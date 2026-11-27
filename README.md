# Grounded SQL repair agent

Grounded repairs SQLite queries for DASE7506 MP3. It uses the schema, cached repair hints and execution feedback from the official starter to generate and check candidate queries before submitting one. It runs on a Linux CPU.

## Run with the official starter

Download and extract the DASE7506 MP3 starter, which supplies the benchmark data, predictor and scorer. [Official starter download](https://xudongwu-0.github.io/courses/dase7506/mp3/downloads/MP3_SQL_student_package.zip?v=406d69507ccf)

Start in the root of this release. Set `STARTER` to the extracted directory containing `mp3_sql/`. The commands use Python 3.12 and write fresh results to a temporary directory.

```bash
RELEASE="$PWD"
STARTER=/absolute/path/to/MP3_SQL_student_package
RUN_DIR=$(mktemp -d)
python3.12 -m venv "$RUN_DIR/venv"
source "$RUN_DIR/venv/bin/activate"
python -m pip install -r "$STARTER/requirements.txt" -r "$RELEASE/requirements.txt"

cd "$STARTER"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RELEASE/submission"
python -m mp3_sql.check_data
for split in validation test; do
  python -m mp3_sql.predict --split "$split" --workers 1 \
    --method "$RELEASE/submission/submission.py" \
    --checkpoint "$RELEASE/submission/config.json" --output "$RUN_DIR/$split.json"
  python -m mp3_sql.score --split "$split" --workers 1 \
    --predictions "$RUN_DIR/$split.json" --output "$RUN_DIR/$split-score.json"
done
echo "Results in $RUN_DIR"
```

## Success rates

The official scorer replays each tool trace. A task passes when the submitted SQL produces the required result on all three grading database instances.

| Evaluation | Tasks passed | Success rate |
| --- | ---: | ---: |
| TRAIN, three-fold out-of-fold by database | 411 / 480 | 85.63% |
| Validation | 101 / 120 | 84.17% |
| Public test | 200 / 240 | 83.33% |

Each TRAIN task was evaluated once with assets fitted on the other two folds. Validation and test use the shipped checkpoint and assets fitted on all 480 TRAIN tasks. The original official test predictions and scorer output are in `outputs/`.
