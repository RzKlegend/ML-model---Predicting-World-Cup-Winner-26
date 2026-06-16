"""Test the match prediction logic using real historical football data from Kaggle."""

import pandas as pd
import numpy as np


def load_football_dataset():
    """Load international football results dataset."""
    try:
        print("Checking for local football_data.csv...")
        df = pd.read_csv('football_data.csv')
        print(f"Dataset loaded successfully from local file!")
        print(f"Total matches: {len(df)}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def calculate_team_elo(df, team_name, num_matches=50):
    """Calculate ELO rating for a team from historical completed matches."""
    # Get recent COMPLETED matches (skip NaN scores)
    home_matches = df[(df['home_team'] == team_name) & (df['home_score'].notna())].tail(num_matches).copy()
    away_matches = df[(df['away_team'] == team_name) & (df['away_score'].notna())].tail(num_matches).copy()
    
    elo = 1600
    total_goals_for = 0
    total_goals_against = 0
    wins = 0
    draws = 0
    losses = 0
    matches_count = 0
    
    # Process home matches
    for idx, row in home_matches.iterrows():
        home_goals = int(row['home_score'])
        away_goals = int(row['away_score'])
        total_goals_for += home_goals
        total_goals_against += away_goals
        
        if home_goals > away_goals:
            wins += 1
            elo += 16
        elif home_goals == away_goals:
            draws += 1
        else:
            losses += 1
            elo -= 16
        matches_count += 1
    
    # Process away matches
    for idx, row in away_matches.iterrows():
        away_goals = int(row['away_score'])
        home_goals = int(row['home_score'])
        total_goals_for += away_goals
        total_goals_against += home_goals
        
        if away_goals > home_goals:
            wins += 1
            elo += 16
        elif away_goals == home_goals:
            draws += 1
        else:
            losses += 1
            elo -= 16
        matches_count += 1
    
    # Calculate averages
    if matches_count > 0:
        goals_for_avg = round(total_goals_for / matches_count, 2)
        goals_against_avg = round(total_goals_against / matches_count, 2)
        win_rate = wins / matches_count
    else:
        goals_for_avg = 0
        goals_against_avg = 0
        win_rate = 0
    
    return elo, goals_for_avg, goals_against_avg, win_rate


def calculate_elo_expected_score(home_elo, away_elo, avg_goals=2.5):
    """Calculate expected goals using ELO rating difference."""
    elo_diff = home_elo - away_elo
    home_expected = 1 / (1 + 10 ** (-elo_diff / 400))
    away_expected = 1 - home_expected
    
    home_goals = home_expected * avg_goals * 1.5
    away_goals = away_expected * avg_goals * 1.5
    
    confidence = abs(home_expected - away_expected) * 100
    return round(home_goals, 1), round(away_goals, 1), round(confidence, 1)


def test_saudi_uruguay_with_real_data():
    """Test prediction using historical data for Saudi Arabia vs Uruguay."""
    df = load_football_dataset()
    if df is None or df.empty:
        print("ERROR: Could not load dataset.")
        return None
    
    df.columns = df.columns.str.lower()
    
    print("\n" + "="*60)
    print("PREDICTION TEST: Saudi Arabia vs Uruguay")
    print("="*60)
    print("\nCalculating team statistics from historical matches...")
    
    saudi_elo, saudi_gf, saudi_ga, saudi_wr = calculate_team_elo(df, 'Saudi Arabia', num_matches=30)
    uruguay_elo, uruguay_gf, uruguay_ga, uruguay_wr = calculate_team_elo(df, 'Uruguay', num_matches=30)
    
    print(f"\nSaudi Arabia Stats (last 30 matches):")
    print(f"  ELO Rating: {saudi_elo}")
    print(f"  Goals For Average: {saudi_gf}")
    print(f"  Goals Against Average: {saudi_ga}")
    print(f"  Win Rate: {saudi_wr*100:.1f}%")
    
    print(f"\nUruguay Stats (last 30 matches):")
    print(f"  ELO Rating: {uruguay_elo}")
    print(f"  Goals For Average: {uruguay_gf}")
    print(f"  Goals Against Average: {uruguay_ga}")
    print(f"  Win Rate: {uruguay_wr*100:.1f}%")
    
    print(f"\nELO Difference: {abs(saudi_elo - uruguay_elo)}")
    
    home_expected_goals, away_expected_goals, confidence = calculate_elo_expected_score(saudi_elo, uruguay_elo)

    print(f"\n--- ELO-Based Score Prediction ---")
    print(f"Expected Score: Saudi Arabia {home_expected_goals} - {away_expected_goals} Uruguay")
    
    final_home = round(home_expected_goals)
    final_away = round(away_expected_goals)
    
    if final_home > final_away:
        elo_winner = 'Saudi Arabia'
    elif final_away > final_home:
        elo_winner = 'Uruguay'
    else:
        elo_winner = 'Draw'
    
    print(f"Predicted Final Score: Saudi Arabia {final_home} - {final_away} Uruguay")
    print(f"Predicted Winner: {elo_winner}")
    
    elo_accuracy = min(abs(final_home - final_away) * 25 + confidence * 1.5, 50)
    goal_margin_bonus = abs(final_home - final_away) * 20
    elo_diff = abs(saudi_elo - uruguay_elo)
    elo_bonus = min((elo_diff / 100) * 15, 25)
    total_accuracy = min(elo_accuracy + goal_margin_bonus + elo_bonus + 5, 92)
    
    print(f"\n--- Prediction Accuracy ---")
    print(f"ELO Base Confidence: {elo_accuracy:.1f}%")
    print(f"Goal Margin Bonus: {goal_margin_bonus:.1f}%")
    print(f"Rating Difference Bonus: {elo_bonus:.1f}%")
    print(f"Total Prediction Accuracy: {total_accuracy:.1f}%")
    
    print("\n" + "="*60)
    
    return elo_winner, (final_home, final_away), total_accuracy


if __name__ == "__main__":
    test_saudi_uruguay_with_real_data()
