"""
generate_data.py
----------------
Generates a realistic synthetic IPL ball-by-ball dataset.
Run this ONCE to create the CSV before launching the app.
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_MATCHES      = 200
TEAMS            = ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
                     "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
                     "Rajasthan Royals", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans"]

BATSMEN          = ["Rohit Sharma", "Virat Kohli", "MS Dhoni", "KL Rahul", "Shubman Gill",
                     "Hardik Pandya", "Rishabh Pant", "Suryakumar Yadav", "David Warner",
                     "Jos Buttler", "Faf du Plessis", "Quinton de Kock", "Ishan Kishan",
                     "Shreyas Iyer", "Sanju Samson", "Prithvi Shaw", "Mayank Agarwal",
                     "Ruturaj Gaikwad", "Devdutt Padikkal", "Manish Pandey"]

BOWLERS          = ["Jasprit Bumrah", "Yuzvendra Chahal", "Rashid Khan", "Trent Boult",
                     "Kagiso Rabada", "Pat Cummins", "Bhuvneshwar Kumar", "Sunil Narine",
                     "Dwayne Bravo", "Ravindra Jadeja", "Axar Patel", "Imran Tahir",
                     "Mohammed Shami", "Deepak Chahar", "Sam Curran", "Harshal Patel",
                     "T Natarajan", "Prasidh Krishna", "Arshdeep Singh", "Kuldeep Yadav"]

VENUES           = ["Wankhede Stadium, Mumbai", "M. Chinnaswamy Stadium, Bangalore",
                     "Eden Gardens, Kolkata", "MA Chidambaram Stadium, Chennai",
                     "Arun Jaitley Stadium, Delhi", "Rajiv Gandhi Intl Cricket Stadium, Hyderabad"]

SEASONS          = list(range(2016, 2024))

# ── Helper functions ───────────────────────────────────────────────────────────

def simulate_innings(match_id, batting_team, bowling_team, venue, season,
                     innings_num, target=None):
    """Simulate one T20 innings ball by ball and return a list of row dicts."""
    rows = []
    total_balls    = 120   # 20 overs × 6 balls
    wickets_fallen = 0
    runs_scored    = 0
    ball_num       = 0

    # Random team strength modifier (0.8 – 1.2)
    team_strength  = random.uniform(0.8, 1.2)

    for over in range(20):
        if wickets_fallen >= 10:
            break
        for ball in range(1, 7):
            if wickets_fallen >= 10:
                break

            ball_num += 1
            balls_remaining = total_balls - ball_num

            # Situational pressure
            if target:
                runs_needed        = target - runs_scored
                required_run_rate  = (runs_needed / (balls_remaining + 1)) * 6
            else:
                required_run_rate  = 8.0

            # Probability of a wicket increases under pressure
            wicket_prob = 0.05 + (0.02 if required_run_rate > 12 else 0) \
                               + (0.03 if wickets_fallen > 7 else 0)
            is_wicket   = random.random() < wicket_prob * (1 / team_strength)

            # Runs on this ball
            if is_wicket:
                ball_runs = 0
                wickets_fallen += 1
            else:
                weights    = [25, 30, 20, 5, 10, 5, 5]   # dot,1,2,3,4,0(wd),6
                outcomes   = [0, 1, 2, 3, 4, 0, 6]
                idx        = random.choices(range(7), weights=weights)[0]
                ball_runs  = outcomes[idx]
                # Boost scoring in death overs
                if over >= 16 and random.random() < 0.35 * team_strength:
                    ball_runs = random.choice([4, 6])

            runs_scored += ball_runs

            current_run_rate   = (runs_scored / ball_num) * 6
            runs_left_val      = (target - runs_scored) if target else max(0, 150 - runs_scored)
            balls_left_val     = balls_remaining
            wickets_left_val   = 10 - wickets_fallen

            # Win flag (only meaningful in 2nd innings)
            if target:
                if runs_scored >= target:
                    result = 1
                elif balls_left_val == 0 or wickets_fallen >= 10:
                    result = 0
                else:
                    result = None   # mid-innings — fill in later
            else:
                result = None

            rows.append({
                "match_id"          : match_id,
                "season"            : season,
                "venue"             : venue,
                "batting_team"      : batting_team,
                "bowling_team"      : bowling_team,
                "innings"           : innings_num,
                "over"              : over + 1,
                "ball"              : ball,
                "batsman"           : random.choice(BATSMEN),
                "bowler"            : random.choice(BOWLERS),
                "runs_off_bat"      : ball_runs,
                "is_wicket"         : int(is_wicket),
                "total_runs"        : runs_scored,
                "wickets"           : wickets_fallen,
                "runs_left"         : runs_left_val,
                "balls_left"        : balls_left_val,
                "wickets_left"      : wickets_left_val,
                "current_run_rate"  : round(current_run_rate, 2),
                "required_run_rate" : round(required_run_rate, 2) if target else 0.0,
                "target"            : target if target else 0,
                "result"            : result,
            })

            # If batting team just won, stop the innings
            if target and runs_scored >= target:
                break

    # ── Fill result for 2nd-innings rows ─────────────────────────────────────
    if target:
        final_result = 1 if runs_scored >= target else 0
        for r in rows:
            if r["result"] is None:
                r["result"] = final_result

    return rows, runs_scored


def generate_dataset():
    all_rows = []
    for match_id in range(1, NUM_MATCHES + 1):
        season      = random.choice(SEASONS)
        venue       = random.choice(VENUES)
        team_a, team_b = random.sample(TEAMS, 2)

        # First innings
        rows_1, target = simulate_innings(match_id, team_a, team_b,
                                          venue, season, innings_num=1)
        target += 1   # target = 1st innings score + 1

        # Second innings
        rows_2, _ = simulate_innings(match_id, team_b, team_a,
                                     venue, season, innings_num=2, target=target)

        all_rows.extend(rows_1)
        all_rows.extend(rows_2)

    df = pd.DataFrame(all_rows)

    # Drop rows with NaN result (only from innings=1 where result wasn't set)
    df = df.dropna(subset=["result"])
    df["result"] = df["result"].astype(int)

    return df


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating IPL dataset …")
    df = generate_dataset()
    out_path = "data/ipl_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset saved → {out_path}")
    print(f"Shape : {df.shape}")
    print(df.head(3).to_string())
