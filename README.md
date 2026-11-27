# DASE7506 MP3 · Joint SQL Repair Agent

## Usage

Run the agent on Linux with a CPU and Python 3.11 or 3.12. Download and extract the [official MP3 student package](https://xudongwu-0.github.io/courses/dase7506/mp3/). Set `MP3_STARTER` to the extracted directory containing `mp3_sql/`.

Run the following commands from the directory containing this README.

```bash
MP3_STARTER="/absolute/path/to/MP3_SQL_student_package"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/submission:$MP3_STARTER"
python -m mp3_sql.check_data
```

To evaluate the validation set, run the commands below. Set `mp3_split=test` to evaluate the test set. New predictions and scores are written to `runs/`.

```bash
mp3_split=validation
mkdir -p runs
python -m mp3_sql.predict --split "$mp3_split" --workers 1 \
  --method submission/submission.py --checkpoint submission/config.json \
  --output "runs/$mp3_split.json"
python -m mp3_sql.score --split "$mp3_split" --workers 1 \
  --predictions "runs/$mp3_split.json" --output "runs/$mp3_split-score.json"
```

## Results

A task counts as correct when the submitted SQL returns the required result on all three grading database instances.

| Split | Correct / total | Accuracy |
| --- | ---: | ---: |
| Training (database-held-out folds) | 442 / 480 | 92.08% |
| Validation | 102 / 120 | 85.00% |
| Test | 197 / 240 | 82.08% |

The released configuration is the one used for validation and test. The original test [predictions and tool traces](outputs/test.json) and [official score](outputs/test-score.json) are included.
