# aytogermany

aytogermany solves the perfect match problem from the German RTL+ show *Are You the One*.

## Installation and use
The packet manager [`uv`](https://docs.astral.sh/uv/) is used for this project.
To download all the necessary packages, call
```shell
uv sync
```

Run the programm via `main.py`. Available seasons are: `"normalo2020","normalo2021", "normalo2022", "normalo2023",  "normalo2024",   "normalo2025", "normalo2026", "vip2021", "vip2022",  "vip2023", "vip2024", "vip2025",  "vip2026"`. By default all seasons constraints are used, set the flag `--end` to compute the results until a chosen week, the weeks are indexed starting from 0.

```shell
# All seasons constraints are used
uv run main.py vip2026 

# Use information from up to week 3 
uv run main.py vip2026 --end 3

# Print out further information about the generated solutions
uv run main.py vip2026 --analyze 

# Plot the match probabilities
uv run main.py vip2026 --plot
```