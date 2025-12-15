import os
import json
import csv
import multiprocessing
from tqdm import tqdm


def print_dict(d: dict):
    for key, value in d.items():
        print(f"{key}: {value}")


def read_json(path):
    # Try to read as a standard JSON file
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except:
        # Fallback: read as JSON Lines (one JSON object per line)
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(e)
                    print(line)
                    continue
        return data


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_csv(path, header=None):
    # If the CSV has a header row, leave header=None; otherwise, provide a list of field names
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, fieldnames=header)
        data = [row for row in reader]
        return data


def multiprocess_worker(text_data, save_path, func, num_processes=1, save_jsonl=False):
    """
    text_data: list of inputs to `func`
    save_path: str path where output will be saved. Note: this will error if the file already exists
    func: function that processes one element of text_data and returns the processed element
    num_processes: int number of CPU cores to use
    save_jsonl: bool, whether to save output in JSONL format
    """
    print(f"Using {num_processes} cores")
    cnt = 0
    cnt_all = 0
    assert not os.path.exists(save_path), "File already exists!"
    save_file = open(save_path, 'w', encoding='utf-8')
    save_results = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(text_data)) as pbar:
            for result in pool.imap_unordered(func, text_data):
                if result is not None:
                    cnt += 1
                    if save_jsonl:
                        save_file.write(json.dumps(result, ensure_ascii=False))
                        save_file.write("\n")
                        save_file.flush()
                    else:
                        save_results.append(result)
                else:
                    print("Result is None")
                cnt_all += 1
                pbar.set_description(f"Processed {cnt} items ({cnt / len(text_data) * 100:.2f}%)")
                pbar.update()

        tqdm.write(f"Processed {cnt} items ({cnt / len(text_data) * 100:.2f}%) in total")

    if not save_jsonl:
        save_json(save_results, save_path)
    else:
        save_file.close()

    print("-------------------DONE-------------------")
