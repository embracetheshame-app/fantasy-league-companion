# Fantasy League Overview and Newsletter Preparation (Summer 2025)

## 🎯 Objective

Arlen wants to build a comprehensive fantasy‑league companion: a tool that can pull historic and current information about each Sleeper league and generate newsletter‑style content on demand.  The ultimate goal is to learn about league history, managers’ tendencies, and how each team stacks up against expert consensus so that the app can produce recaps, power‑rankings and fun facts automatically.  This report collects the raw data, explains how to access it and highlights key insights for the **2025 season** and earlier.

## 🔗 How data was gathered

* **Sleeper API:**  Sleeper provides unauthenticated, read‑only endpoints to retrieve league information, rosters and users.  Each league JSON object includes important details—like scoring settings, roster positions and a `previous_league_id`—which allows us to traverse the league’s history【29297104568813†L25-L33】.  Rosters and users are fetched separately【54793957898166†L42-L49】【630624547543919†L60-L63】.
* **Historical linkage:**  For each league we followed the `previous_league_id` pointer until it returned `null`.  This reveals the chain of seasons, letting us compare scoring rules and team line‑ups over time.  Example: the 2025 **Fangtasy Football** league (ID 1180243200866377728) links back to its 2024 league (ID 1049192869187264512) via `previous_league_id`【29297104568813†L31-L33】.
* **External rankings:**  Most fantasy ranking sites are JavaScript‑heavy.  Instead of live scraping, we plan to integrate a publicly available dataset.  The **Fantasy Football Data Pros** (FFDP) API exposes projections and historical fantasy points back to 1999【133276187627379†L0-L29】.  These numbers can serve as an objective baseline to rate players.  Because FFDP’s projections are for 2020, they are slightly dated, but they illustrate how to incorporate external rankings.

> **Note about restrictions:**  In this environment, direct HTTP requests from Python are blocked.  All API calls were performed via the browser tool.  When implementing the app locally, you can use libraries like `requests` to call the Sleeper and FFDP APIs directly.

## 🏆 League summaries

The tables below provide essential details for each of Arlen’s leagues.  Scoring settings shown here are the most consequential (QB passing yards, TDs, receptions, etc.).  They illustrate how different leagues reward players, which is vital when comparing rosters or projecting outcomes.

### Live Good – Die Nasty (League ID 1180229588687335424)

| Season | Status | Roster slots (starters/bench) | Key scoring settings | Previous league | Notes |
|---|---|---|---|---|---|
| **2025** | In season | QB, RB, RB, WR, WR, TE, FLEX, **SUPER FLEX**, DEF, 8 bench【238541101318620†L16-L27】 | Pass TD = 4 pt, Rush TD = 6 pt, Reception = 0.5 pt, Pass YD = 0.04 pt【238541101318620†L16-L27】 | 2024 (1048476433129070592)【238541101318620†L32-L34】 | 12 teams; half‑PPR scoring.  Super‑flex encourages multiple QBs. |
| **2024** | Completed | Same roster structure | Same scoring settings【806572407091648†L34-L37】 | 2023 (916389331307397120) | Final standings available via the app. |
| **2023** | Completed | Same roster structure | Same scoring settings【560185604105277†L34-L36】 | 2022 (813110485628112896) | Draft data accessible. |
| **2022** | Completed | Same roster structure | Same scoring settings【299666994383794†L32-L33】 | *None* | Earliest recorded season; baseline for historical comparisons. |

**Insights:**  The league uses **half‑PPR** scoring and awards four points per passing touchdown.  With a super‑flex spot, quarterbacks are extremely valuable.  Because the scoring settings have remained stable across seasons, managers can be compared directly.  The roster endpoint reveals each team’s players and starters【54793957898166†L42-L49】.  User information shows display names and team names for all 12 managers【630624547543919†L60-L63】.

### Cle2Bay (League ID 1180663573516509184)

| Season | Status | Roster slots | Key scoring settings | Previous league | Notes |
|---|---|---|---|---|---|
| **2025** | In season | QB, RB, RB, WR, WR, TE, FLEX, DEF, 9 bench | Pass TD = 4 pt, Rush TD = 6 pt, Reception = 1 pt, Pass YD ≈ 0.039 pt | 2024 (1056323109512478720)【197597754092153†L35-L36】 | 10 teams; full‑PPR scoring, making pass‑catching RBs more valuable. |
| **2024** | Completed | Same roster structure | Same scoring settings | 2023 (934294672728403968) | |
| **2023** | Completed | Same roster structure | Same scoring settings | 2022 (823331049625780224) | |
| **2022** | Completed | Same roster structure | Same scoring settings | 2021 (734904637324402688)【423245499815562†L36-L37】 | Earliest recorded season. |

**Insights:**  Cle2Bay uses **full‑PPR** scoring.  Managers who stack wide receivers or pass‑catching running backs tend to finish higher.  Historical data will help identify if certain managers consistently favor PPR specialists.

### The Collective (League ID 1180396624690155520)

| Season | Status | Roster slots | Key scoring settings | Previous league | Notes |
|---|---|---|---|---|---|
| **2025** | In season | QB, RB, RB, WR, WR, TE, FLEX, DEF, 8 bench | Pass TD = 4 pt, Rush TD = 6 pt, Reception = 1 pt | 2024 (1048285051798134784)【245113043670204†L33-L34】 | 10 teams; full‑PPR. |
| **2024** | Completed | Same roster structure | Same scoring settings | 2023 (978086811614420992)【901878869457539†L33-L34】 | |
| **2023** | Completed | Same roster structure | Same scoring settings | 2022 (872245588312375296)【901441438843370†L31-L32】 | |
| **2022** | Completed | Same roster structure | Same scoring settings | *None* | Earliest season—foundation for historical analysis. |

**Insights:**  The Collective has a consistent rule set across seasons and also uses full‑PPR scoring.  Looking at each manager’s draft strategies over time will reveal whether they adapt to the PPR format (e.g., prioritizing slot receivers and tight ends with high target volumes).

### Sidekickers (League ID 1181662601757753344)

| Season | Status | Roster slots | Key scoring settings | Previous league | Notes |
|---|---|---|---|---|---|
| **2025** | In season | QB, RB, RB, WR, WR, TE, FLEX, SUPER FLEX, DEF, 7 bench | Pass TD = 4 pt, Rush TD = 6 pt, Reception = 1 pt | 2024 (1124072579184021504)【51893415053877†L30-L31】 | 10 teams; includes super‑flex and full‑PPR. |
| **2024** | Completed | Same roster structure | Same scoring settings【112121901126838†L31-L34】 | 2023 (916389262462001152) | |
| **2023** | Completed | Same roster structure | Same scoring settings【112121901126838†L31-L34】 | 2022 (863915777030549504) | |
| **2022** | Completed | Same roster structure | Same scoring settings【112121901126838†L31-L34】 | *None* | Earliest season. |

**Insights:**  Like Live Good – Die Nasty, Sidekickers allows a **super‑flex**, increasing the demand for quarterbacks.  Full‑PPR scoring further boosts pass‑catchers.  Historical rosters show which managers exploit the super‑flex by rostering multiple high‑scoring QBs.

### Fangtasy Football (League ID 1180243200866377728)

| Season | Status | Roster slots | Key scoring settings | Previous league | Notes |
|---|---|---|---|---|---|
| **2025** | Pre‑draft | QB, RB, RB, WR, WR, TE, FLEX, 8 bench【29297104568813†L31-L34】 | Pass TD = 4 pt, Rush TD = 6 pt, Reception = 0.5 pt (half‑PPR), bonus points for 300/400 pass yd games【29297104568813†L13-L23】 | 2024 (1049192869187264512)【29297104568813†L31-L33】 | 10 teams; includes bonus scoring for large passing games.  Draft yet to occur as of July 31, 2025. |
| **2024** | Completed | Same roster structure | Same scoring settings【346814421895097†L15-L24】 | *None* | Final standings available. |

**Insights:**  Fangtasy Football is a smaller league (10 teams) with half‑PPR scoring and **bonus points** for 300‑ and 400‑yard passing games【346814421895097†L15-L24】.  Because the league is preparing for the 2025 draft, historical data from 2024 will inform draft strategies.  Since there is no previous league before 2024, this league is relatively new.

## 👥 Manager profiles and tendencies (concept)

Sleeper user data lists each manager’s `display_name`, `user_id` and team name【630624547543919†L60-L63】.  To build manager profiles:

1. **Aggregate historical rosters**:  For each manager, fetch rosters from every season in which they participated by calling `/league/{league_id}/rosters`.  The `owner_id` field links roster entries to a user.  By compiling all players drafted or acquired, we can identify position preferences (e.g., RB‑heavy vs. WR‑heavy) and risk tolerance (taking rookies vs. veterans).
2. **Evaluate transaction history**:  Use `/league/{league_id}/transactions/{week}` to review waiver moves and trades.  Managers who trade frequently might be more aggressive; those who pick up specific types of players indicate certain strategies.
3. **Compute fantasy points**:  For each player on a manager’s roster, pull fantasy points from FFDP’s historical endpoints (e.g., `api/players/{year}/all` for season totals【133276187627379†L0-L29】).  Summing these for all players reveals whether the manager consistently drafts high‑scoring players or relies on breakout candidates.
4. **Detect consistency across leagues**:  Many managers play in multiple leagues.  Use `user/{user_id}/leagues/nfl/{season}` (Sleeper endpoint) to list all leagues for each year and cross‑reference strategies.

By automating these steps, the app can generate a short paragraph describing each manager’s tendencies—e.g., “**larsont85** consistently rosters two elite quarterbacks and targets high‑volume wide receivers, resulting in top‑3 finishes in 2023 and 2024.”  Since such analysis relies on aggregated historical data, the app should recompute these profiles at the beginning of each season.

## 🔍 Comparing rosters to consensus rankings

To evaluate team strength and forecast the season, we need a way to compare each roster with expert rankings.  Because major ranking sites do not expose open APIs, we suggest two approaches:

1. **Use FFDP projections as a ranking baseline:**  The `/api/projections` endpoint from Fantasy Football Data Pros lists projected fantasy points for each player (2020 season)【133276187627379†L0-L29】.  Even though the projections are dated, they illustrate how to obtain a numeric value for every player.  The app can compute a team’s projected score by summing the values of all starters.  This yields a simple “power ranking.”  Replace the 2020 projections with up‑to‑date numbers (from your own model or a paid service) for more accurate results.
2. **Allow CSV import of expert rankings:**  During the preseason, download a consensus ranking file (e.g., from FantasyPros) and upload it into the app.  The app can map Sleeper player IDs to ranking positions and show the difference between a team’s roster and the consensus draft board.  This method ensures the latest data without violating site policies.

## 📰 Newsletter structure and content ideas

The app should be able to automatically produce various types of content:

* **Season kickoff newsletter:**  Includes a recap of last season’s final standings, notable transactions, draft steals, and projections for the upcoming season.  Highlight managers’ tendencies and fun facts (e.g., longest winning streak, highest single‑game score).  Summarize rule changes and how they affect strategy.
* **Weekly recap:**  Summarize the highest‑scoring team and biggest upset, call out close finishes, list top free‑agent pickups, and provide a short preview of next week’s match‑ups.
* **Manager spotlight:**  Each edition can feature a profile of one manager, including their historical record, favorite types of players, memorable trades, and predictions.

When generating text, keep the tone fun yet professional, reflecting Arlen’s preferred style of concise bullet points followed by short paragraphs.

## 🛠️ App architecture (proposed)

1. **Backend:**  Use Python with the `requests` library to call Sleeper’s API (for leagues, rosters, users, matchups and transactions) and the FFDP API for projections.  Build helper functions to follow the `previous_league_id` chain and aggregate data.  Cache responses locally to avoid repeated API calls.
2. **Data storage:**  Store league and player data in JSON or a lightweight database (e.g., SQLite).  Historical data can be stored permanently so you can compute manager tendencies across seasons without refetching.
3. **Frontend:**  Build a simple web app with **Streamlit**.  The app should allow Arlen to:
   * Select one of his leagues via a drop‑down (listing league names and IDs).
   * View league settings, roster breakdowns and user information.
   * Compare team rosters against consensus rankings by summing projection scores.
   * Generate newsletter reports with one click (season recap, weekly recap, manager spotlight).
4. **Deployment:**  Deploy the Streamlit app to a free platform like Streamlit Cloud or host it on your own server.  With minor modifications you can package it as a React Native or Flutter app for mobile app stores.

## ✅ Next steps

1. **Gather complete historical rosters and transaction logs** for each league using the Sleeper API endpoints.  This will fuel the manager profiles and power rankings.
2. **Decide on a projection source:**  Use FFDP’s projection endpoint for a proof of concept.  For more accurate 2025 numbers, consider licensing a data feed or manually importing a CSV from FantasyPros.
3. **Implement the app** using the provided `fantasy_app.py` template (see file below).  Test the app locally and refine the UI and newsletter generator.

When the app is ready, Arlen will have a powerful tool to monitor his leagues, evaluate team strengths and automatically craft newsletters and posts throughout the season.