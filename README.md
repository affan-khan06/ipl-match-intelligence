# 🏏 IPL Match Intelligence System

A data science web app built with Streamlit that predicts IPL match win probability, analyses player performance, generates insights, and lets you simulate match scenarios interactively.

---

## 🚀 Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset  (~200 matches, ball-by-ball)
python generate_data.py

# 3. Train the ML model
python train_model.py

# 4. Launch the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📂 Project Structure

```
ipl_intelligence/
├── app.py               ← Main Streamlit app  (all pages + UI)
├── generate_data.py     ← Synthetic IPL dataset generator
├── train_model.py       ← Model training & evaluation script
├── requirements.txt     ← Python dependencies
├── README.md
├── data/
│   └── ipl_data.csv     ← Generated after running generate_data.py
└── models/
    └── win_probability_model.pkl   ← Saved after running train_model.py
```

---

## 🧩 Using a Real Dataset

If you have a real IPL ball-by-ball CSV, place it at `data/ipl_data.csv`.
The app expects these columns:

| Column | Description |
|--------|-------------|
| `match_id` | Unique match ID |
| `season` | Year (e.g. 2023) |
| `venue` | Ground name |
| `batting_team` | Team name |
| `bowling_team` | Team name |
| `innings` | 1 or 2 |
| `over` | 1–20 |
| `ball` | 1–6 |
| `batsman` | Player name |
| `bowler` | Player name |
| `runs_off_bat` | Runs scored on this ball |
| `is_wicket` | 1 if wicket fell, else 0 |
| `total_runs` | Cumulative team runs |
| `wickets` | Wickets fallen so far |
| `runs_left` | Runs needed to win (2nd innings) |
| `balls_left` | Balls remaining |
| `wickets_left` | Wickets remaining |
| `current_run_rate` | CRR at this ball |
| `required_run_rate` | RRR at this ball |
| `target` | Target set by 1st innings |
| `result` | 1 = batting team wins, 0 = loses |

Popular public datasets:
- [Cricsheet](https://cricsheet.org/downloads/) (YAML format, needs parsing)
- [Kaggle IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)

---

## 🧠 How the Model Works

| Item | Detail |
|------|--------|
| Algorithm | Random Forest (200 trees, depth 8) |
| Features | `runs_left`, `balls_left`, `wickets_left`, `current_run_rate`, `required_run_rate` |
| Target | `result` (1 = batting team wins) |
| Output | Win probability 0–1 |
| Typical accuracy | 75–85% on synthetic data |

---

## 📄 Splitting into Modules (next step)

For a production codebase, split `app.py` into:

```
ipl_intelligence/
├── app.py                     ← Streamlit entry, page routing only
├── pages/
│   ├── win_predictor.py       ← Win Predictor page
│   ├── player_stats.py        ← Player Stats page
│   ├── insight_engine.py      ← Insight Engine page
│   └── match_simulator.py     ← Match Simulator page
├── core/
│   ├── data_loader.py         ← load_data(), compute_player_stats()
│   ├── model.py               ← load_model(), predict_win_probability()
│   ├── insights.py            ← generate_insights()
│   └── charts.py              ← all Plotly figure functions
├── generate_data.py
├── train_model.py
└── requirements.txt
```

---

## 🔮 Future Enhancements

- [ ] Live match integration via Cricbuzz / ESPN Cricinfo API
- [ ] Deep learning model (LSTM for ball sequence)
- [ ] Bowler performance analytics page
- [ ] Mobile-responsive layout
- [ ] Commentary generator ("Bumrah to Kohli, beaten outside off …")
- [ ] Docker / cloud deployment (Streamlit Community Cloud)

---

## 📝 License

MIT — free to use, modify, and distribute.
