Each parser provides a function, `get_paths`, which gets a dataframe of the following format:
| last_name | first_name | birth_year | death_year | title | path | split_num | midi_parser | duration |
| --------- | ---------- | ---------- | ---------- | ----- | ---- | --------- | ----------- | -------- |
| Chopin | Frédéric | 1810 | 1849 | Ballade No.4, Op.52 | Chopin, Frédéric, Ballade No.4, Op.52, 7tmQSWuYwrI-split-1.mid | 1 | main | 30 |
| Chopin | Frédéric | 1810 | 1849 | Ballade No.4, Op.52 | Chopin, Frédéric, Ballade No.4, Op.52, 7tmQSWuYwrI-split-2.mid | 2 | main | 30 |

## Structure

- splitter.py has an interface to split midis (called by common.py)
- common.py has a function to generate the appropriate dataframes
- src folder actually uses the output of common.py
