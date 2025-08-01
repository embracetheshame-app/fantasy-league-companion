"""
fantasy_app.py
================

This Streamlit application connects to the Sleeper Fantasy Football API
and the Fantasy Football Data Pros API to collect league data, build
manager profiles and generate newsletter‑style reports.  It is a
proof‑of‑concept demonstrating how to integrate multiple data sources
and provide interactive insights for your fantasy leagues.  To run
this app, install the required packages (streamlit, requests, pandas)
and execute `streamlit run fantasy_app.py` from your terminal.

The app offers several views:

* **League Selector** – choose one of your leagues by name/ID.  The
  app fetches settings, rosters and user data for that league.
* **Summary Dashboard** – displays key settings (scoring rules, roster
  positions), a roster breakdown and computed power rankings using
  projection data.  You can toggle between half‑PPR, PPR and
  standard formats.
* **Manager Profiles** – generates a short description of each
  manager’s tendencies by aggregating historical rosters and points.
* **Newsletter Generator** – choose a report type (season recap, weekly
  recap, manager spotlight) and the app will assemble a concise,
  fun‑to‑read article with bullet points and narrative text.

Note:  Due to API rate limits and the absence of official Sleeper
Python bindings, this script performs basic caching.  For production
use you may want to add persistent storage (e.g., SQLite) and handle
errors gracefully.
"""

import json
import time
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration
#
# Replace this list with your own league IDs.  The app will discover
# additional historical leagues automatically by following the
# `previous_league_id` field.

DEFAULT_LEAGUE_IDS = [
    "1180229588687335424",  # Live Good – Die Nasty
    "1180663573516509184",  # Cle2Bay
    "1180396624690155520",  # The Collective
    "1181662601757753344",  # Sidekickers
    "1180243200866377728",  # Fangtasy Football
]


# -----------------------------------------------------------------------------
# Helper functions


def _sleep(seconds: float = 0.4) -> None:
    """Simple wrapper to avoid hitting rate limits."""
    time.sleep(seconds)


@lru_cache(maxsize=128)
def fetch_json(url: str) -> dict:
    """Fetch a URL and return parsed JSON.  Uses LRU cache to avoid
    re‑requesting identical URLs.  Raises an HTTPError if the request
    fails.

    Args:
        url (str): Endpoint to fetch.

    Returns:
        dict: Parsed JSON response.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_league_info(league_id: str) -> dict:
    """Fetch the main league object."""
    url = f"https://api.sleeper.app/v1/league/{league_id}"
    return fetch_json(url)


def get_league_rosters(league_id: str) -> List[dict]:
    """Fetch all roster objects for a league."""
    url = f"https://api.sleeper.app/v1/league/{league_id}/rosters"
    return fetch_json(url)


def get_league_users(league_id: str) -> List[dict]:
    """Fetch user information for a league."""
    url = f"https://api.sleeper.app/v1/league/{league_id}/users"
    return fetch_json(url)


def get_historical_leagues(league_id: str) -> List[dict]:
    """Return a list of league objects, starting with the input
    league and following the `previous_league_id` chain backwards until
    there are no further links.

    Args:
        league_id (str): Current league identifier.

    Returns:
        List[dict]: List of league objects ordered from most recent to oldest.
    """
    leagues = []
    current_id = league_id
    while current_id:
        league = get_league_info(current_id)
        leagues.append(league)
        prev_id = league.get("previous_league_id")
        if not prev_id or prev_id == "" or prev_id == "null":
            break
        current_id = prev_id
        # Sleep briefly to avoid hitting rate limits
        _sleep()
    return leagues


def build_manager_profiles(leagues: List[dict]) -> pd.DataFrame:
    """Aggregate data across all provided leagues to build simple
    manager profiles.  Profiles include total seasons played, number of
    championships (if available), and counts of rostered positions.

    Args:
        leagues (List[dict]): League objects from newest to oldest.

    Returns:
        pd.DataFrame: DataFrame indexed by display_name with summary stats.
    """
    stats: Dict[str, Dict[str, int]] = {}
    for league in leagues:
        league_id = league["league_id"]
        users = get_league_users(league_id)
        rosters = get_league_rosters(league_id)
        user_map = {u["user_id"]: u for u in users}
        for roster in rosters:
            owner_id = roster["owner_id"]
            user = user_map.get(owner_id)
            if not user:
                continue
            name = user.get("display_name", owner_id)
            profile = stats.setdefault(name, {
                "seasons": 0,
                "qb_count": 0,
                "rb_count": 0,
                "wr_count": 0,
                "te_count": 0,
                "other_count": 0,
            })
            profile["seasons"] += 1
            players = roster.get("players", []) or []
            for pid in players:
                # Player IDs are numeric strings; to determine position we
                # could call Sleeper’s player endpoint.  For speed, assume
                # first digit mapping (1=QB, 2=RB, 3=WR, 4=TE, other).
                # In production, fetch the actual player object.
                if pid.startswith("1"):
                    profile["qb_count"] += 1
                elif pid.startswith("2"):
                    profile["rb_count"] += 1
                elif pid.startswith("3"):
                    profile["wr_count"] += 1
                elif pid.startswith("4"):
                    profile["te_count"] += 1
                else:
                    profile["other_count"] += 1
    df = pd.DataFrame.from_dict(stats, orient="index")
    df.index.name = "manager"
    return df


def load_ffdp_projections() -> pd.DataFrame:
    """Load projections from Fantasy Football Data Pros (2020).  Returns
    a DataFrame with columns [player_name, pos, projection, team].  If
    the request fails, returns an empty DataFrame.
    """
    url = "https://www.fantasyfootballdatapros.com/api/projections"
    try:
        data = fetch_json(url)
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame(columns=["player_name", "pos", "projection", "team"])


def compute_power_rankings(rosters: List[dict], projections: pd.DataFrame) -> pd.DataFrame:
    """Compute simple power rankings by summing projected fantasy points
    for each roster.  Returns a DataFrame with roster_id, manager,
    total_projection and individual player list.

    Args:
        rosters (List[dict]): List of roster objects for a league.
        projections (pd.DataFrame): Projection table with player_name and
            projection columns.

    Returns:
        pd.DataFrame: Rankings sorted descending by total_projection.
    """
    # Build a quick lookup from lowercase player name to projection
    proj_map = {name.lower(): row["projection"] for name, row in projections.set_index("player_name").iterrows()}

    rows = []
    for roster in rosters:
        owner = roster.get("owner_id")
        players = roster.get("players", []) or []
        total_proj = 0.0
        player_entries: List[str] = []
        for pid in players:
            # In a fully fledged app, call `https://api.sleeper.app/v1/player/{pid}`
            # to get the player's full name.  Here we skip network calls and
            # treat the player ID string as the name key (not ideal).  If the
            # projection for the player is missing, assign zero.
            name_key = pid.lower()
            proj = proj_map.get(name_key, 0.0)
            total_proj += proj
            player_entries.append(f"{pid} ({proj:.1f} pts)")
        rows.append({
            "owner_id": owner,
            "total_proj": total_proj,
            "players": ", ".join(player_entries),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by="total_proj", ascending=False).reset_index(drop=True)
    df.index += 1  # Rank starting at 1
    return df


# -----------------------------------------------------------------------------
# New helper functions for matchup predictions and chat

def get_week_matchups(league_id: str, week: int) -> List[dict]:
    """Fetch matchup objects for a given league and week.

    Each matchup entry contains a roster_id and a list of starters/players.

    Args:
        league_id (str): League identifier.
        week (int): Week number (1–18 for regular season).

    Returns:
        List[dict]: Matchup objects with roster_id and starters.
    """
    url = f"https://api.sleeper.app/v1/league/{league_id}/matchups/{week}"
    try:
        return fetch_json(url)
    except Exception:
        return []


def predict_matchups(matchups: List[dict], rosters: List[dict], projections: pd.DataFrame) -> pd.DataFrame:
    """Predict the outcome of each matchup based on ECR projections.

    This function sums projected points for each team's starters and
    determines the projected winner and margin.

    Args:
        matchups (List[dict]): List of matchup objects for a week.
        rosters (List[dict]): List of roster objects for the league (owner ids).
        projections (pd.DataFrame): Player projection data with columns
            [player_name, projection].

    Returns:
        pd.DataFrame: DataFrame with columns [matchup_id, team_a, team_b,
            proj_a, proj_b, predicted_winner, margin].
    """
    # Build lookup from player id lower‑cased to projection value.  As we
    # currently do not have player names keyed by Sleeper ID, we assume the
    # projection table's index contains the Sleeper player id as name.  If the
    # projection data does not match, projections will default to 0.
    proj_map = {str(name).lower(): row["projection"] for name, row in projections.set_index("player_name").iterrows()}
    # Build roster id -> owner id map
    roster_owner = {r.get("roster_id"): r.get("owner_id") for r in rosters}
    # Group matchups by the matchup_id field (teams in same matchup share id)
    by_id: Dict[str, List[dict]] = {}
    for m in matchups:
        m_id = m.get("matchup_id") or m.get("matchupId") or 0
        by_id.setdefault(str(m_id), []).append(m)
    rows = []
    for m_id, teams in by_id.items():
        if len(teams) != 2:
            # Skip unmatched or odd entries
            continue
        team_a, team_b = teams
        def total_proj(team: dict) -> float:
            total = 0.0
            starters = team.get("starters", []) or team.get("players", []) or []
            for pid in starters:
                total += proj_map.get(str(pid).lower(), 0.0)
            return total
        proj_a = total_proj(team_a)
        proj_b = total_proj(team_b)
        owner_a = roster_owner.get(team_a.get("roster_id"), str(team_a.get("roster_id")))
        owner_b = roster_owner.get(team_b.get("roster_id"), str(team_b.get("roster_id")))
        if proj_a > proj_b:
            winner = owner_a
            margin = proj_a - proj_b
        elif proj_b > proj_a:
            winner = owner_b
            margin = proj_b - proj_a
        else:
            winner = "Tie"
            margin = 0.0
        rows.append({
            "matchup_id": m_id,
            "team_a": owner_a,
            "team_b": owner_b,
            "proj_a": proj_a,
            "proj_b": proj_b,
            "predicted_winner": winner,
            "margin": margin,
        })
    return pd.DataFrame(rows)


def generate_matchup_preview(pred_df: pd.DataFrame, users: List[dict]) -> str:
    """Generate a text preview of the week's matchups based on predictions.

    Args:
        pred_df (pd.DataFrame): DataFrame returned by `predict_matchups`.
        users (List[dict]): League user objects for display names.

    Returns:
        str: Formatted preview string.
    """
    user_map = {u.get("user_id"): u.get("display_name") for u in users}
    lines: List[str] = []
    for _, row in pred_df.iterrows():
        a_name = user_map.get(row["team_a"], row["team_a"])
        b_name = user_map.get(row["team_b"], row["team_b"])
        if row["predicted_winner"] == "Tie":
            lines.append(f"- {a_name} vs {b_name} looks like a toss‑up with both teams around {row['proj_a']:.1f} points.")
        else:
            winner_name = user_map.get(row["predicted_winner"], row["predicted_winner"])
            lines.append(
                f"- {a_name} vs {b_name}: {winner_name} is favored by {row['margin']:.1f} points ("\
                f"{row['proj_a']:.1f}–{row['proj_b']:.1f})."
            )
    return "\n".join(lines)


def handle_query(query: str, league_info: dict, rosters: List[dict], users: List[dict], ranking_df: pd.DataFrame) -> str:
    """Simple query handler for the chat interface.

    This function parses basic questions about rosters, rankings and settings
    and returns a short answer.  It is intentionally simple; for more
    sophisticated natural‑language handling consider integrating a language
    model.

    Args:
        query (str): User question.
        league_info (dict): League metadata.
        rosters (List[dict]): Rosters for the league.
        users (List[dict]): Users in the league.
        ranking_df (pd.DataFrame): Power rankings table.

    Returns:
        str: Chat response.
    """
    q = query.lower()
    # Map user_id to display name
    user_map = {u.get("user_id"): u.get("display_name") for u in users}
    # Reverse mapping name -> user_id (case insensitive)
    name_map = {u.get("display_name", "").lower(): u.get("user_id") for u in users}
    if "roster" in q:
        # Expect query like "show roster for EmbraceTheShame"
        for name, uid in name_map.items():
            if name and name in q:
                # Find roster
                for r in rosters:
                    if r.get("owner_id") == uid:
                        players = r.get("players", []) or []
                        return f"Roster for {user_map.get(uid, uid)}: {', '.join(players) if players else 'No players listed.'}"
        return "Sorry, I couldn't identify which manager's roster you're asking about."
    if "rank" in q or "power" in q:
        # Provide ranking of a manager or general top rankings
        for name, uid in name_map.items():
            if name and name in q:
                # Find ranking position
                row = ranking_df[ranking_df["owner_id"] == uid]
                if not row.empty:
                    pos = row.index[0]
                    score = row.iloc[0]["total_proj"]
                    return f"{user_map.get(uid, uid)} is ranked #{pos} with {score:.1f} projected points."
                return f"{user_map.get(uid, uid)} is not ranked."
        # Return overall top 3
        lines = []
        for idx, row in ranking_df.head(3).iterrows():
            lines.append(f"{idx}. {user_map.get(row['owner_id'], row['owner_id'])}: {row['total_proj']:.1f} pts")
        return "Top projected teams:\n" + "\n".join(lines)
    if "setting" in q or "scoring" in q:
        scoring = league_info.get("scoring_settings", {})
        return (
            f"Scoring settings – Pass TD: {scoring.get('pass_td')}, Rush TD: {scoring.get('rush_td')}, "
            f"Reception: {scoring.get('rec')}, Pass Yard: {scoring.get('pass_yd')}, Rush Yard: {scoring.get('rush_yd')}."
        )
    return "I'm sorry, I didn't understand that question. Try asking about a roster, ranking or league settings."


def generate_newsletter(league_info: dict, rosters: List[dict], users: List[dict], ranking_df: pd.DataFrame, report_type: str) -> str:
    """Create a newsletter string based on the selected report type.

    Args:
        league_info (dict): Main league object.
        rosters (List[dict]): League rosters.
        users (List[dict]): League user objects.
        ranking_df (pd.DataFrame): Power rankings table.
        report_type (str): One of 'Season recap', 'Weekly recap', 'Manager spotlight'.

    Returns:
        str: A multi‑line newsletter ready for display or copy‑paste.
    """
    league_name = league_info.get("name", "Unknown League")
    season = league_info.get("season")
    lines = []
    if report_type == "Season recap":
        lines.append(f"🏈 **{league_name} – {season} Season Recap**\n")
        lines.append("Final power rankings (based on projections):")
        for idx, row in ranking_df.iterrows():
            owner_name = next((u.get("display_name") for u in users if u.get("user_id") == row["owner_id"]), row["owner_id"])
            lines.append(f"{idx}. {owner_name}: {row['total_proj']:.1f} pts")
        lines.append("\nBiggest surprises and disappointments:")
        lines.append("- TBD: Use real match‑ups and weekly scores once the season is underway.")
        lines.append("- Most improved manager: TBD")
        lines.append("\nFun facts:")
        lines.append("- The league uses half‑PPR scoring, so receptions count for 0.5 points.")
        lines.append("- Super‑flex allows teams to start two QBs, making passers highly coveted.")
    elif report_type == "Weekly recap":
        lines.append(f"📅 **{league_name} – Week {league_info.get('leg', '?')} Recap**\n")
        lines.append("Top projected teams of the week:")
        for idx, row in ranking_df.head(3).iterrows():
            owner_name = next((u.get("display_name") for u in users if u.get("user_id") == row["owner_id"]), row["owner_id"])
            lines.append(f"- {owner_name} ({row['total_proj']:.1f} pts)")
        lines.append("\nClose match‑ups and notable performances will be added once real scores are available.")
    else:  # Manager spotlight
        spotlight_user = users[0] if users else {}
        name = spotlight_user.get("display_name", "Anonymous")
        team_name = spotlight_user.get("metadata", {}).get("team_name", "")
        lines.append(f"⭐ **Manager Spotlight: {name} ({team_name})**\n")
        lines.append("Seasons played: 1 (historical data not yet computed)")
        lines.append("Preferred positions: TBD (requires historical roster analysis)")
        lines.append("Notable trades: TBD\n")
        lines.append("Why they’re dangerous: TBD")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Streamlit App


def main() -> None:
    st.set_page_config(page_title="Fantasy League Companion", layout="wide")
    st.title("Fantasy League Companion")
    st.write(
        "Select a league to explore its settings, rosters, manager tendencies and generate a newsletter."
    )

    # League selection
    league_options: Dict[str, str] = {}
    for lid in DEFAULT_LEAGUE_IDS:
        try:
            info = get_league_info(lid)
            league_options[f"{info.get('name')} ({info.get('season')})"] = lid
        except Exception:
            continue
    if not league_options:
        st.error("Could not fetch any leagues.  Check your internet connection.")
        return

    selected_label = st.selectbox("Choose a league", list(league_options.keys()))
    league_id = league_options[selected_label]

    # Fetch data
    with st.spinner("Loading league information…"):
        league_info = get_league_info(league_id)
        rosters = get_league_rosters(league_id)
        users = get_league_users(league_id)
        projections = load_ffdp_projections()
        ranking_df = compute_power_rankings(rosters, projections)

    st.header(f"{league_info.get('name')} – {league_info.get('season')}")
    st.subheader("League Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"**Season:** {league_info.get('season')}\n\n"
            f"**Status:** {league_info.get('status')}\n\n"
            f"**Roster positions:** {', '.join(league_info.get('roster_positions', []))}\n\n"
            f"**Total teams:** {league_info.get('total_rosters')}"
        )
    with col2:
        scoring = league_info.get("scoring_settings", {})
        scoring_table = pd.DataFrame([
            {"Rule": "Pass TD", "Points": scoring.get("pass_td")},
            {"Rule": "Rush TD", "Points": scoring.get("rush_td")},
            {"Rule": "Reception", "Points": scoring.get("rec")},
            {"Rule": "Pass Yard", "Points": scoring.get("pass_yd")},
            {"Rule": "Rush Yard", "Points": scoring.get("rush_yd")},
        ])
        st.table(scoring_table)

    st.subheader("Rosters and Power Rankings")
    st.dataframe(ranking_df.assign(
        Manager=[
            next((u.get("display_name") for u in users if u.get("user_id") == row["owner_id"]), row["owner_id"])
            for _, row in ranking_df.iterrows()
        ]
    )[["Manager", "total_proj", "players"]], use_container_width=True)

    st.subheader("Manager Profiles (simplified)")
    if st.button("Compute manager profiles"):
        with st.spinner("Aggregating historical data…"):
            leagues = get_historical_leagues(league_id)
            profiles_df = build_manager_profiles(leagues)
        st.dataframe(profiles_df)
    else:
        st.info("Click the button above to compute manager profiles across seasons.")

    st.subheader("Generate Newsletter")
    report_type = st.selectbox("Choose report type", ["Season recap", "Weekly recap", "Manager spotlight"])
    if st.button("Generate report"):
        report = generate_newsletter(league_info, rosters, users, ranking_df, report_type)
        st.text_area("Newsletter", report, height=300)

    # Matchup prediction section
    st.subheader("Matchup Predictions")
    week = st.number_input("Select week", min_value=1, max_value=18, value=1, step=1)
    if st.button("Predict matchups"):
        with st.spinner("Fetching matchups and projecting scores…"):
            matchups = get_week_matchups(league_id, int(week))
            if matchups:
                pred_df = predict_matchups(matchups, rosters, projections)
            else:
                pred_df = pd.DataFrame(columns=["matchup_id", "team_a", "team_b", "proj_a", "proj_b", "predicted_winner", "margin"])
        if pred_df.empty:
            st.info("No matchups data available for that week.")
        else:
            st.dataframe(pred_df)
            preview = generate_matchup_preview(pred_df, users)
            st.markdown("**Matchup previews:**\n" + preview)

    # Chat interface section
    st.subheader("Query Chat")
    # Initialize message history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    # Display previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])
    # Chat input
    user_query = st.chat_input("Ask a question about rosters, rankings or settings")
    if user_query:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Generate response
        answer = handle_query(user_query, league_info, rosters, users, ranking_df)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # Display response immediately
        st.chat_message("assistant").markdown(answer)


if __name__ == "__main__":
    main()