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
try:
    import openai  # Optional, used for LLM‑based newsletters
except ImportError:
    openai = None


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
    for each roster using real player names and projections.

    This function loads the full Sleeper player mapping (cached) to
    translate player IDs into names.  It then normalizes the names
    (lowercased, stripped of punctuation) to match against the
    projections table from Fantasy Football Data Pros.  Each team's
    projected score is the sum of its players' projected points.

    Args:
        rosters (List[dict]): List of roster objects for a league.
        projections (pd.DataFrame): Projection table with columns
            [player_name, projection].

    Returns:
        pd.DataFrame: Rankings sorted descending by total_proj with
            columns [owner_id, total_proj, players].
    """
    # Build a lookup from normalized player name to projection value
    # (e.g. "christianmccaffrey" -> 22.5).  Normalize once to
    # accelerate lookup in the loops below.
    proj_map: Dict[str, float] = {}
    for _, row in projections.iterrows():
        name: str = row.get("player_name", "")
        if not name:
            continue
        key = normalize_player_name(name)
        # If multiple entries exist (duplicate names), keep the max
        proj_map[key] = max(proj_map.get(key, 0.0), row.get("projection", 0.0))

    # Load player metadata from Sleeper to map IDs to names and positions
    players_meta = load_sleeper_players()

    rows = []
    for roster in rosters:
        owner = roster.get("owner_id")
        players = roster.get("players", []) or []
        total_proj = 0.0
        player_entries: List[str] = []
        for pid in players:
            pid_str = str(pid)
            meta = players_meta.get(pid_str, {}) or {}
            full_name = meta.get("full_name") or meta.get("name") or pid_str
            key = normalize_player_name(full_name)
            proj = proj_map.get(key, 0.0)
            total_proj += proj
            player_entries.append(f"{full_name} ({proj:.1f} pts)")
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
    """Predict the outcome of each matchup using real player names and projections.

    This function sums projected points for each team's starters
    (preferring the `starters` list on the matchup object; if not
    present, it falls back to all players on the roster).  It also
    records the top projected players for each team to aid in the
    narrative preview.  Player IDs are translated to names via the
    Sleeper players mapping and normalized to match against the
    projection table.

    Args:
        matchups (List[dict]): List of matchup objects for a week.
        rosters (List[dict]): List of roster objects for the league (owner ids).
        projections (pd.DataFrame): Player projection data with columns
            [player_name, projection].

    Returns:
        pd.DataFrame: DataFrame with columns [matchup_id, team_a, team_b,
            proj_a, proj_b, predicted_winner, margin, key_players_a,
            key_players_b].  The key_players_x columns are lists of the
            top projected players (by name) for that team.
    """
    # Build normalized projection map: name -> projection
    proj_map: Dict[str, float] = {}
    for _, row in projections.iterrows():
        pname: str = row.get("player_name", "")
        if not pname:
            continue
        key = normalize_player_name(pname)
        proj_map[key] = max(proj_map.get(key, 0.0), row.get("projection", 0.0))
    # Load player metadata
    players_meta = load_sleeper_players()
    # Map roster_id to owner_id
    roster_owner = {r.get("roster_id"): r.get("owner_id") for r in rosters}
    # Group matchups by matchup_id (each should have exactly two teams)
    by_id: Dict[str, List[dict]] = {}
    for m in matchups:
        m_id = m.get("matchup_id") or m.get("matchupId") or 0
        by_id.setdefault(str(m_id), []).append(m)
    rows = []
    for m_id, teams in by_id.items():
        if len(teams) != 2:
            continue
        team_a, team_b = teams
        def compute_score_and_keys(team: dict) -> (float, List[str]):
            total = 0.0
            starters = team.get("starters") or team.get("players") or []
            key_players: List[tuple] = []  # (name, projection)
            for pid in starters:
                pid_str = str(pid)
                meta = players_meta.get(pid_str, {}) or {}
                full_name = meta.get("full_name") or meta.get("name") or pid_str
                norm = normalize_player_name(full_name)
                points = proj_map.get(norm, 0.0)
                total += points
                key_players.append((full_name, points))
            # sort by projection descending and take top 3
            top_names = [name for name, _ in sorted(key_players, key=lambda x: x[1], reverse=True)[:3]]
            return total, top_names
        proj_a, key_a = compute_score_and_keys(team_a)
        proj_b, key_b = compute_score_and_keys(team_b)
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
            "key_players_a": key_a,
            "key_players_b": key_b,
        })
    return pd.DataFrame(rows)


def generate_matchup_preview(pred_df: pd.DataFrame, users: List[dict]) -> str:
    """Generate a narrative preview of the week's matchups.

    This function converts internal user IDs to display names and
    constructs storylines for each matchup.  It highlights the
    projected margin and key players on each team.  Ties are
    presented as close battles.  If key player data is not
    available, it falls back gracefully.

    Args:
        pred_df (pd.DataFrame): DataFrame returned by `predict_matchups`.
        users (List[dict]): League user objects for display names.

    Returns:
        str: Formatted preview string with bullet points.
    """
    user_map = {u.get("user_id"): u.get("display_name") for u in users}
    lines: List[str] = []
    for _, row in pred_df.iterrows():
        a_name = user_map.get(row["team_a"], row["team_a"])
        b_name = user_map.get(row["team_b"], row["team_b"])
        key_a = row.get("key_players_a", [])
        key_b = row.get("key_players_b", [])
        if row["predicted_winner"] == "Tie":
            # Toss‑up: both teams projected similarly
            key_blurb = " and ".join([
                f"{a_name} leans on {', '.join(key_a)}" if key_a else "",
                f"{b_name} counters with {', '.join(key_b)}" if key_b else "",
            ]).strip()
            description = f"looks like a toss‑up around {row['proj_a']:.1f} points"
            if key_blurb:
                description += f" – {key_blurb}."
            lines.append(f"- {a_name} vs {b_name} {description}")
        else:
            winner_name = user_map.get(row["predicted_winner"], row["predicted_winner"])
            loser_name = b_name if row["predicted_winner"] == row["team_a"] else a_name
            winner_keys = key_a if row["predicted_winner"] == row["team_a"] else key_b
            loser_keys = key_b if winner_keys is key_a else key_a
            desc = f"is favored by {row['margin']:.1f} points "
            desc += f"({row['proj_a']:.1f}–{row['proj_b']:.1f}). "
            # Highlight star players
            if winner_keys:
                desc += f"{winner_name} will count on {', '.join(winner_keys)}"
                if loser_keys:
                    desc += f", while {loser_name} hopes {', '.join(loser_keys)} can keep it close."
                else:
                    desc += "."
            lines.append(f"- {a_name} vs {b_name}: {winner_name} {desc}")
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
    if "lineup" in q or "start" in q:
        # Suggest lineup for a manager
        for name, uid in name_map.items():
            if name and name in q:
                roster = next((r for r in rosters if r.get("owner_id") == uid), None)
                if roster:
                    lineup = suggest_starting_lineup(roster, league_info, load_ffdp_projections())
                    return f"Lineup advice for {user_map.get(uid, uid)}:\n{lineup}"
                return f"Couldn't find roster for {user_map.get(uid, uid)}."
        return "Please specify which manager you'd like a lineup suggestion for."
    if "trade" in q:
        # Suggest trade partner for a manager
        pos_totals = compute_positional_totals(rosters, load_ffdp_projections())
        for name, uid in name_map.items():
            if name and name in q:
                suggestion = suggest_trade_partner(uid, pos_totals, users)
                return suggestion or "No suitable trade partner found."
        return "Please specify which manager you're asking about for a trade suggestion."
    if "strength" in q or "weakness" in q:
        # Provide strengths/weaknesses for a manager
        pos_totals = compute_positional_totals(rosters, load_ffdp_projections())
        for name, uid in name_map.items():
            if name and name in q:
                mgr_row = pos_totals[pos_totals["owner_id"] == uid]
                if mgr_row.empty:
                    return f"No data found for {user_map.get(uid, uid)}."
                mgr_row = mgr_row.iloc[0]
                league_avg = pos_totals[["QB", "RB", "WR", "TE"]].mean()
                strengths = [pos for pos in ["QB", "RB", "WR", "TE"] if mgr_row[pos] > league_avg[pos]]
                weaknesses = [pos for pos in ["QB", "RB", "WR", "TE"] if mgr_row[pos] < league_avg[pos]]
                return f"{user_map.get(uid, uid)}'s strengths: {', '.join(strengths) if strengths else 'None'}; weaknesses: {', '.join(weaknesses) if weaknesses else 'None'}."
        return "Specify which manager you want strength/weakness information for."
    return "I'm sorry, I didn't understand that question. Try asking about a roster, ranking or league settings."


# -----------------------------------------------------------------------------
# Optional LLM‑based chat handler

def handle_query_llm(
    query: str,
    league_info: dict,
    rosters: List[dict],
    users: List[dict],
    ranking_df: pd.DataFrame,
    fallback_handler: callable = None,
) -> str:
    """Handle a chat query using a language model if available.

    This helper attempts to generate a richer, more nuanced answer to
    questions about the league by calling the OpenAI API.  It first
    checks for the availability of the `openai` module and an API
    key (via `st.secrets['openai_api_key']`).  If either is missing,
    it falls back to the provided `fallback_handler` (typically
    `handle_query`).  The prompt includes a summary of the league's
    name, season, top projected teams and scoring settings to give
    the model context.  If the API call fails for any reason, the
    fallback handler is used.

    Args:
        query (str): The user query from the chat box.
        league_info (dict): Current league metadata.
        rosters (List[dict]): Roster objects (unused directly here).
        users (List[dict]): User objects for display names.
        ranking_df (pd.DataFrame): Current power rankings.
        fallback_handler (callable, optional): Function to call on
            failure. Defaults to `handle_query` if None.

    Returns:
        str: Response generated by the language model or fallback.
    """
    if fallback_handler is None:
        fallback_handler = handle_query
    # Ensure API key and openai are available
    api_key = None
    if hasattr(st, "secrets"):
        api_key = st.secrets.get("openai_api_key")
    # If no LLM available, use fallback
    if openai is None or not api_key:
        return fallback_handler(query, league_info, rosters, users, ranking_df)
    # Prepare context
    try:
        openai.api_key = api_key
        league_name = league_info.get("name", "Your league")
        season = league_info.get("season", "this season")
        # Build a list of top teams for context
        top_lines = []
        for idx, row in ranking_df.head(5).iterrows():
            manager_name = next(
                (u.get("display_name") for u in users if u.get("user_id") == row["owner_id"]),
                row["owner_id"],
            )
            top_lines.append(f"#{idx} {manager_name} ({row['total_proj']:.1f} pts)")
        # Summarize scoring settings
        scoring = league_info.get("scoring_settings", {}) or {}
        rec_pts = scoring.get("rec") or 0
        rec_desc = "half‑PPR" if rec_pts == 0.5 else ("PPR" if rec_pts == 1 else "standard")
        scoring_desc = (
            f"Pass TD {scoring.get('pass_td', '?')} pts, Rush TD {scoring.get('rush_td', '?')} pts, "
            f"Reception {rec_pts} pts ({rec_desc})."
        )
        # Compose messages for the chat completion
        system_context = (
            "You are a fantasy football assistant. Answer questions about a Sleeper "
            "fantasy league concisely and helpfully using the provided context."
        )
        league_context = (
            f"League name: {league_name}, season: {season}. Top projected teams: {', '.join(top_lines)}. "
            f"Scoring: {scoring_desc}"
        )
        messages = [
            {"role": "system", "content": system_context},
            {"role": "system", "content": league_context},
            {"role": "user", "content": query},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.6,
            max_tokens=300,
        )
        reply = response.choices[0].message.get("content", "").strip()
        if reply:
            return reply
        return fallback_handler(query, league_info, rosters, users, ranking_df)
    except Exception:
        return fallback_handler(query, league_info, rosters, users, ranking_df)


# -----------------------------------------------------------------------------
# Sleeper player mapping and normalization helpers

@lru_cache(maxsize=1)
def load_sleeper_players() -> Dict[str, dict]:
    """Load the full Sleeper players dictionary.

    Returns:
        Dict[str, dict]: Mapping of player_id to player metadata.
    """
    url = "https://api.sleeper.app/v1/players/nfl"
    try:
        return fetch_json(url)
    except Exception:
        return {}


def normalize_player_name(name: str) -> str:
    """Normalize a player's name for matching.

    Lowercases, removes spaces and punctuation.
    """
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower()) if name else ""


# -----------------------------------------------------------------------------
# Analysis helpers for manager strengths, trade suggestions and lineup advice

def compute_positional_totals(rosters: List[dict], projections: pd.DataFrame) -> pd.DataFrame:
    """Compute the total projected points by position for each roster.

    This helper returns a DataFrame where each row corresponds to a
    roster and columns include `owner_id`, `QB`, `RB`, `WR`, `TE` and
    `Other`.  Totals are computed using projections from the provided
    projections table and the Sleeper player metadata.

    Args:
        rosters (List[dict]): Roster objects for a league.
        projections (pd.DataFrame): Projection data with columns
            [player_name, projection].

    Returns:
        pd.DataFrame: DataFrame with positional totals.
    """
    # Build normalized projection map
    proj_map: Dict[str, float] = {}
    for _, row in projections.iterrows():
        pname = row.get("player_name", "")
        if not pname:
            continue
        key = normalize_player_name(pname)
        proj_map[key] = max(proj_map.get(key, 0.0), row.get("projection", 0.0))
    players_meta = load_sleeper_players()
    result_rows: List[dict] = []
    for roster in rosters:
        owner = roster.get("owner_id")
        pos_totals = {"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0, "Other": 0.0}
        for pid in roster.get("players", []) or []:
            pid_str = str(pid)
            meta = players_meta.get(pid_str, {}) or {}
            pos = meta.get("position") or "Other"
            full_name = meta.get("full_name") or meta.get("name") or pid_str
            key = normalize_player_name(full_name)
            pts = proj_map.get(key, 0.0)
            if pos not in pos_totals:
                pos = "Other"
            pos_totals[pos] += pts
        pos_totals["owner_id"] = owner
        result_rows.append(pos_totals)
    df = pd.DataFrame(result_rows)
    # Ensure missing columns appear
    for col in ["QB", "RB", "WR", "TE", "Other"]:
        if col not in df.columns:
            df[col] = 0.0
    return df[["owner_id", "QB", "RB", "WR", "TE", "Other"]]


def suggest_trade_partner(owner_id: str, pos_totals: pd.DataFrame, users: List[dict]) -> Optional[str]:
    """Suggest a potential trade partner for the given manager.

    The heuristic identifies the manager's weakest position and finds
    another team that is strong at that position but weak where the
    manager is strong.  If no complementary partner exists, returns
    None.

    Args:
        owner_id (str): User ID of the manager.
        pos_totals (pd.DataFrame): Position totals by owner_id.
        users (List[dict]): User objects for display names.

    Returns:
        Optional[str]: A human‑readable suggestion like
            "Consider trading with {partner_name}: you need RB help and they need WRs".
    """
    # Compute league averages for each position (excluding owner column)
    stat_cols = ["QB", "RB", "WR", "TE"]
    league_avg = pos_totals[stat_cols].mean()
    # Get manager row
    mgr_row = pos_totals[pos_totals["owner_id"] == owner_id]
    if mgr_row.empty:
        return None
    mgr_row = mgr_row.iloc[0]
    # Differences from league average
    diffs = {col: mgr_row[col] - league_avg[col] for col in stat_cols}
    # Identify strongest and weakest positions for manager
    strength_pos = max(diffs, key=diffs.get)
    weakness_pos = min(diffs, key=diffs.get)
    # Build user map for names
    name_map = {u.get("user_id"): u.get("display_name") for u in users}
    best_partner = None
    best_score = -float("inf")
    for _, row in pos_totals.iterrows():
        other_id = row["owner_id"]
        if other_id == owner_id:
            continue
        # this team should be strong at manager's weakness and weak at manager's strength
        diff_weak = row[weakness_pos] - league_avg[weakness_pos]
        diff_strength = row[strength_pos] - league_avg[strength_pos]
        # require they have a positive surplus in weakness_pos and a negative in strength_pos
        score = diff_weak - diff_strength
        if diff_weak > 0 and diff_strength < 0 and score > best_score:
            best_score = score
            best_partner = other_id
    if best_partner is None:
        return None
    partner_name = name_map.get(best_partner, str(best_partner))
    return f"Consider trading with {partner_name}: you need help at {weakness_pos}, and they could use {strength_pos}."


def suggest_starting_lineup(roster: dict, league_info: dict, projections: pd.DataFrame) -> str:
    """Suggest an optimal starting lineup for a given roster based on projections.

    This function examines the league's roster positions to determine
    how many starters are required at each position (QB, RB, WR, TE,
    FLEX and SUPER_FLEX).  It then selects the highest projected
    players available on the team to fill those slots.  Remaining
    players are considered bench.  Returns a formatted string with
    recommendations.

    Args:
        roster (dict): Roster object for one team.
        league_info (dict): League settings including roster_positions.
        projections (pd.DataFrame): Projection data with player_name and projection.

    Returns:
        str: A human‑readable lineup suggestion.
    """
    # Build normalized projection map
    proj_map: Dict[str, float] = {}
    for _, row in projections.iterrows():
        pname = row.get("player_name", "")
        if not pname:
            continue
        key = normalize_player_name(pname)
        proj_map[key] = max(proj_map.get(key, 0.0), row.get("projection", 0.0))
    players_meta = load_sleeper_players()
    # Determine starting slot counts
    roster_positions = league_info.get("roster_positions", []) or []
    # Count starting requirements excluding bench and other non‑start slots
    from collections import Counter
    pos_counts = Counter()
    for pos in roster_positions:
        upper = pos.upper()
        # Skip bench and special slots
        if upper in {"BN", "IR", "TAXI", "PRACTICE", "RES"}:
            continue
        pos_counts[upper] += 1
    # Extract all players with metadata and projections
    player_list: List[tuple] = []  # (name, pos, projection)
    for pid in roster.get("players", []) or []:
        pid_str = str(pid)
        meta = players_meta.get(pid_str, {}) or {}
        full_name = meta.get("full_name") or meta.get("name") or pid_str
        pos = (meta.get("position") or "Other").upper()
        norm = normalize_player_name(full_name)
        proj = proj_map.get(norm, 0.0)
        player_list.append((full_name, pos, proj))
    # Build lineup
    used = set()
    starters: List[str] = []
    # Helper to select players for a given position
    def select_players(position: str, count: int, eligible_positions: List[str]):
        selected = []
        for _ in range(count):
            # Among unused players with eligible positions, pick highest projection
            candidates = [(i, p) for i, p in enumerate(player_list) if i not in used and p[1] in eligible_positions]
            if not candidates:
                continue
            idx, player = max(candidates, key=lambda x: x[1][2])
            used.add(idx)
            selected.append(player)
        return selected
    # Standard positions
    starters += [f"{name} ({proj:.1f} pts)" for name, _, proj in select_players("QB", pos_counts.get("QB", 0), ["QB"])]
    starters += [f"{name} ({proj:.1f} pts)" for name, _, proj in select_players("RB", pos_counts.get("RB", 0), ["RB"])]
    starters += [f"{name} ({proj:.1f} pts)" for name, _, proj in select_players("WR", pos_counts.get("WR", 0), ["WR"])]
    starters += [f"{name} ({proj:.1f} pts)" for name, _, proj in select_players("TE", pos_counts.get("TE", 0), ["TE"])]
    # Flex positions (RB/WR/TE)
    starters += [f"{name} ({proj:.1f} pts)" for name, _, proj in select_players("FLEX", pos_counts.get("FLEX", 0), ["RB", "WR", "TE"])]
    # Super flex positions (QB/RB/WR/TE)
    starters += [f"{name} ({proj:.1f} pts)" for name, _, proj in select_players("SUPER_FLEX", pos_counts.get("SUPER_FLEX", 0), ["QB", "RB", "WR", "TE"])]
    # Bench are remaining players
    bench = [f"{name} ({proj:.1f} pts)" for i, (name, _, proj) in enumerate(player_list) if i not in used]
    # Format recommendation
    output_lines = []
    output_lines.append("**Recommended starters:**")
    if starters:
        for s in starters:
            output_lines.append(f"- {s}")
    else:
        output_lines.append("(No eligible starters found)")
    output_lines.append("\n**Bench options:**")
    if bench:
        for b in bench:
            output_lines.append(f"- {b}")
    else:
        output_lines.append("(No remaining players)")
    return "\n".join(output_lines)


def generate_newsletter(
    league_info: dict,
    rosters: List[dict],
    users: List[dict],
    ranking_df: pd.DataFrame,
    report_type: str,
    spotlight_owner_id: Optional[str] = None,
) -> str:
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
        # Describe scoring settings succinctly
        scoring = league_info.get("scoring_settings", {})
        rec_pts = scoring.get("rec")
        rec_desc = "half‑PPR" if rec_pts == 0.5 else ("PPR" if rec_pts == 1 else "standard")
        lines.append(f"- This is a {rec_desc} league (receptions worth {rec_pts or 0} points).")
        # Describe roster structure
        positions = league_info.get("roster_positions", []) or []
        starter_counts = {pos: positions.count(pos) for pos in set(positions) if pos not in {"BN", "IR", "TAXI", "PRACTICE", "RES"}}
        pos_desc = ", ".join([f"{cnt}×{pos}" for pos, cnt in starter_counts.items()]) if starter_counts else "unknown"
        lines.append(f"- Starting lineup: {pos_desc}. Bench and taxi spots not shown.")
    elif report_type == "Weekly recap":
        lines.append(f"📅 **{league_name} – Week {league_info.get('leg', '?')} Recap**\n")
        lines.append("Top projected teams of the week:")
        for idx, row in ranking_df.head(3).iterrows():
            owner_name = next((u.get("display_name") for u in users if u.get("user_id") == row["owner_id"]), row["owner_id"])
            lines.append(f"- {owner_name} ({row['total_proj']:.1f} pts)")
        lines.append("\nClose match‑ups and notable performances will be added once real scores are available.")
    else:  # Manager spotlight
        # Determine which manager to spotlight.  If a specific owner_id
        # was provided, use it; otherwise default to the first user.
        spotlight_user = None
        owner_id = spotlight_owner_id
        if owner_id:
            spotlight_user = next((u for u in users if u.get("user_id") == owner_id), None)
        if spotlight_user is None and users:
            spotlight_user = users[0]
            owner_id = spotlight_user.get("user_id")
        name = spotlight_user.get("display_name", "Anonymous") if spotlight_user else "Anonymous"
        team_name = spotlight_user.get("metadata", {}).get("team_name", "") if spotlight_user else ""
        lines.append(f"⭐ **Manager Spotlight: {name} ({team_name})**\n")
        # Basic season count: how many rosters does this user have across history?
        # This function only knows the current season; a more robust count
        # would use `build_manager_profiles` and historical leagues.  Here we
        # approximate seasons as 1.
        lines.append("Seasons played: 1 (historical data not yet computed)")
        # Compute positional totals to infer strengths/weaknesses
        pos_totals = compute_positional_totals(rosters, load_ffdp_projections())
        # Suggest trade partner
        suggestion = suggest_trade_partner(owner_id, pos_totals, users)
        lines.append("\nPositional analysis:")
        if not pos_totals.empty:
            mgr_row = pos_totals[pos_totals["owner_id"] == owner_id]
            if not mgr_row.empty:
                mgr_row = mgr_row.iloc[0]
                # Compute league averages
                league_avg = pos_totals[["QB", "RB", "WR", "TE"]].mean()
                # Strengths and weaknesses
                strengths = []
                weaknesses = []
                for pos in ["QB", "RB", "WR", "TE"]:
                    diff = mgr_row[pos] - league_avg[pos]
                    if diff > 0:
                        strengths.append(pos)
                    elif diff < 0:
                        weaknesses.append(pos)
                lines.append(f"- Strengths: {', '.join(strengths) if strengths else 'None' }")
                lines.append(f"- Weaknesses: {', '.join(weaknesses) if weaknesses else 'None' }")
            else:
                lines.append("- No roster data available.")
        else:
            lines.append("- Could not compute positional totals.")
        if suggestion:
            lines.append(f"\nTrade idea: {suggestion}")
        # Starting lineup recommendation
        # Find roster of this manager
        spotlight_roster = next((r for r in rosters if r.get("owner_id") == owner_id), None)
        if spotlight_roster:
            lineup = suggest_starting_lineup(spotlight_roster, league_info, load_ffdp_projections())
            lines.append("\nLineup advice:\n" + lineup)
        else:
            lines.append("\nLineup advice: roster not found.")
    return "\n".join(lines)


def generate_newsletter_llm(
    league_info: dict,
    rosters: List[dict],
    users: List[dict],
    ranking_df: pd.DataFrame,
    report_type: str,
    spotlight_owner_id: Optional[str] = None,
) -> str:
    """Generate a newsletter using an LLM (e.g. ChatGPT).

    This helper constructs a prompt summarizing the league, rankings and
    (optionally) manager spotlight and sends it to the OpenAI API to
    generate a narrative article.  If the OpenAI client is not
    available or no API key is configured, it falls back to the
    internal rule‑based `generate_newsletter`.

    Args:
        league_info (dict): League metadata.
        rosters (List[dict]): Roster objects.
        users (List[dict]): User objects.
        ranking_df (pd.DataFrame): Power rankings.
        report_type (str): Type of report ('Season recap', 'Weekly recap', 'Manager spotlight').
        spotlight_owner_id (Optional[str]): Manager ID for spotlight (if applicable).

    Returns:
        str: Newsletter text.
    """
    # If OpenAI library or API key is missing, use fallback
    api_key = None
    if hasattr(st, "secrets"):
        api_key = st.secrets.get("openai_api_key")
    if openai is None or not api_key:
        return generate_newsletter(
            league_info, rosters, users, ranking_df, report_type, spotlight_owner_id=spotlight_owner_id
        )
    openai.api_key = api_key
    league_name = league_info.get("name", "Unknown League")
    season = league_info.get("season", "")
    # Build a brief summary of power rankings
    rankings_summary = []
    for idx, row in ranking_df.head(6).iterrows():
        manager_name = next((u.get("display_name") for u in users if u.get("user_id") == row["owner_id"]), row["owner_id"])
        rankings_summary.append(f"#{idx} {manager_name} ({row['total_proj']:.1f} pts)")
    # Determine spotlight manager
    spotlight_text = ""
    if report_type == "Manager spotlight":
        # Get selected manager name
        manager = next((u for u in users if u.get("user_id") == spotlight_owner_id), None)
        if manager:
            spotlight_text = f"Focus on manager {manager.get('display_name')} and their team."
    # Construct prompt
    prompt = (
        f"You are a fantasy football analyst writing a {report_type.lower()} for the league '{league_name}' in {season}. "
        f"Here are the top projected teams: {', '.join(rankings_summary)}. "
        f"{spotlight_text}"
        " Write a concise yet engaging newsletter with bullet points and narrative that highlights key storylines, strengths and weaknesses, and fun facts."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful fantasy football newsletter writer."},
                     {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=700,
        )
        content = response.choices[0].message.get("content", "").strip()
        return content or generate_newsletter(
            league_info, rosters, users, ranking_df, report_type, spotlight_owner_id=spotlight_owner_id
        )
    except Exception:
        # Fallback on any error
        return generate_newsletter(
            league_info, rosters, users, ranking_df, report_type, spotlight_owner_id=spotlight_owner_id
        )


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
    # Variables to hold context for newsletter generation
    selected_spotlight_owner_id: Optional[str] = None
    selected_league_for_recap: dict = league_info
    selected_rosters_for_recap: List[dict] = rosters
    selected_users_for_recap: List[dict] = users
    selected_ranking_for_recap: pd.DataFrame = ranking_df
    # When season recap is selected, allow choosing a specific year
    if report_type == "Season recap":
        # Fetch historical leagues once for season selection
        historical = get_historical_leagues(league_id)
        # Build map of season year -> league object
        season_map = {l.get("season"): l for l in historical if l.get("season")}
        # Sort seasons descending (most recent first)
        season_years = sorted(season_map.keys(), reverse=True)
        season_choice = st.selectbox("Select season year", season_years)
        # Update selected league data if different from current
        selected_league_for_recap = season_map.get(season_choice, league_info)
        # If not the same as current, fetch rosters/users for that season
        if selected_league_for_recap.get("league_id") != league_info.get("league_id"):
            rec_league_id = selected_league_for_recap.get("league_id")
            selected_rosters_for_recap = get_league_rosters(rec_league_id)
            selected_users_for_recap = get_league_users(rec_league_id)
            selected_ranking_for_recap = compute_power_rankings(selected_rosters_for_recap, projections)
        else:
            # Same season, reuse existing data
            selected_rosters_for_recap = rosters
            selected_users_for_recap = users
            selected_ranking_for_recap = ranking_df
    # If manager spotlight, allow picking which manager
    if report_type == "Manager spotlight":
        manager_names = {u.get("display_name", u.get("user_id")): u.get("user_id") for u in users}
        spotlight_name = st.selectbox("Choose a manager to spotlight", list(manager_names.keys()))
        selected_spotlight_owner_id = manager_names.get(spotlight_name)
    # Allow user to choose whether to use an LLM to generate the newsletter
    use_llm_newsletter = False
    llm_available = openai is not None and hasattr(st, "secrets") and st.secrets.get("openai_api_key")
    if llm_available:
        use_llm_newsletter = st.checkbox("Use ChatGPT for newsletter", value=False)
    if st.button("Generate report"):
        # Determine generator based on LLM availability and user preference
        if llm_available and use_llm_newsletter:
            report = generate_newsletter_llm(
                selected_league_for_recap,
                selected_rosters_for_recap,
                selected_users_for_recap,
                selected_ranking_for_recap,
                report_type,
                spotlight_owner_id=selected_spotlight_owner_id,
            )
        else:
            report = generate_newsletter(
                selected_league_for_recap,
                selected_rosters_for_recap,
                selected_users_for_recap,
                selected_ranking_for_recap,
                report_type,
                spotlight_owner_id=selected_spotlight_owner_id,
            )
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



    # Manager analysis section
    st.subheader("Manager Analysis")
    # Build mapping of display names to owner_ids
    manager_options = {u.get("display_name", u.get("user_id")): u.get("user_id") for u in users}
    selected_manager_name = st.selectbox("Select a manager to analyze", list(manager_options.keys()))
    selected_owner_id = manager_options.get(selected_manager_name)
    if st.button("Analyze manager"):
        with st.spinner("Computing strengths, trade suggestions and lineup recommendations…"):
            pos_totals = compute_positional_totals(rosters, projections)
            suggestion = suggest_trade_partner(selected_owner_id, pos_totals, users)
            spotlight_roster = next((r for r in rosters if r.get("owner_id") == selected_owner_id), None)
            strengths = []
            weaknesses = []
            # Strengths and weaknesses relative to league averages
            if not pos_totals.empty:
                mgr_row = pos_totals[pos_totals["owner_id"] == selected_owner_id]
                if not mgr_row.empty:
                    mgr_row = mgr_row.iloc[0]
                    league_avg = pos_totals[["QB", "RB", "WR", "TE"]].mean()
                    for pos in ["QB", "RB", "WR", "TE"]:
                        diff = mgr_row[pos] - league_avg[pos]
                        if diff > 0:
                            strengths.append(pos)
                        elif diff < 0:
                            weaknesses.append(pos)
            lineup_advice = "No roster found." if not spotlight_roster else suggest_starting_lineup(spotlight_roster, league_info, projections)
        # Display analysis
        st.markdown(f"**Strengths:** {', '.join(strengths) if strengths else 'None'}")
        st.markdown(f"**Weaknesses:** {', '.join(weaknesses) if weaknesses else 'None'}")
        if suggestion:
            st.markdown(f"**Trade suggestion:** {suggestion}")
        st.markdown("**Lineup recommendation:**")
        st.markdown(lineup_advice)

    # ----------------------------------------------------------------------
    # Chat interface (placed after manager analysis)
    st.subheader("Query Chat")
    # Determine if LLM is available for chat responses
    use_llm_chat = False
    llm_available_chat = openai is not None and hasattr(st, "secrets") and st.secrets.get("openai_api_key")
    if llm_available_chat:
        use_llm_chat = st.checkbox("Use ChatGPT for chat answers", value=False)
    # Initialize message history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    # Display previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])
    # Chat input
    user_query = st.chat_input("Ask a question about rosters, rankings, trades or settings")
    if user_query:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Generate response using LLM or fallback
        if llm_available_chat and use_llm_chat:
            answer = handle_query_llm(user_query, league_info, rosters, users, ranking_df, fallback_handler=handle_query)
        else:
            answer = handle_query(user_query, league_info, rosters, users, ranking_df)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # Display response immediately
        st.chat_message("assistant").markdown(answer)


if __name__ == "__main__":
    main()