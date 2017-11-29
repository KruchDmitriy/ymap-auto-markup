from os import listdir
import json

MARKUP_TASKS = "./data/markup_tasks/"
RESULTS = './data/results.json'

itasks = {}
new_results = []

skip_count = 0
if __name__ == "__main__":
    for filename in listdir(MARKUP_TASKS):
        with open(MARKUP_TASKS + filename, 'r') as file:
            task = json.load(file)
            for item in task:
                if item['id'] not in itasks:
                    itasks[item['id']] = []
                itasks[item['id']].append(filename)

    with open(RESULTS, 'r') as markup_file:
        results = json.load(markup_file)
        for result in results:
            if len(result['results']) == 0:
                continue
            taskId = itasks[result['results'][0]['id']][0]
            new_result = {
                'task': taskId,
                'results': [],
                'user': result['user']
            }
            new_results.append(new_result)
            for current in result['results']:
                if taskId not in itasks[current['id']]:
                    skip_count += 1
                else:
                    new_result['results'].append(current)
    with open(RESULTS + ".new", 'w') as markup_file:
        json.dump(new_results, markup_file)
