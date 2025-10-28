

## Setup

```shell
git clone https://github.com/daraghhollman/messenger_region_classifier.git
cd messenger_region_classifier
```

The required Philpott+ (2020) boundary crossing intervals list is not available for automatic download. Please manually download [supporting_table_S1.tab](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/1U6FEO) as original format (.xlsx) and save to `./data/philpott_2020_crossing_list.xlsx`.

> [!TIP]
> For portability, we **strongly** recommend and use [uv](https://docs.astral.sh/uv/) to manage dependencies and versions. If you do not wish to use uv, dependencies can be found within [pyproject.toml](./pyproject.toml) and a `requirements.txt` file can be made with `pip compile pyproject.toml -o requirements.txt`.

Download the required MESSENGER data and SPICE kernels with:
(requires [wget](https://www.gnu.org/software/wget/))
```shell
chmod +x ./init
./init
```
This will take a long time, and is highly dependent on network and read/write speeds.

## Running

### Data Reduction

```shell
uv run src/reduce_data.py
```
