import os
import json
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

path = './files'

result_dict = {}

for filename in os.listdir(path):
    with open(path + '/' + filename) as file:
        data = json.load(file)

        # check suffix
        type = filename[filename.index('_') + 1:filename.index('.')]

        # get user
        user = ''
        if type == 'demo' or type == 'short':
            user = data[0]['user']
        elif type == 'raw':
            raw_data = json.loads(data[0]['experiment'])
            user = raw_data['userId']

        if user != '' and user not in result_dict:
            result_dict[user] = {type: data}
        elif user != '':
            result_dict[user][type] = data

crossval_arr = []
crossvalavg_arr = []
# write samples in form of a dict with entries iteration, strategy, run, accuracy

filename = 'validation.all (4).json'
with open('./' + filename) as file:
    data = json.load(file)

    for ind, array in enumerate(data['randomAccuracyKSplits']):
        for itr, accuracy in enumerate(array):
            crossval_arr.append({'strategy': 'random', 'iteration': itr, 'run': ind, 'accuracy': accuracy})

    for ind, array in enumerate(data['uncertaintyAccuracyKSplits']):
        for itr, accuracy in enumerate(array):
            crossval_arr.append({'strategy': 'uncertainty', 'iteration': itr, 'run': ind, 'accuracy': accuracy})

    for ind, array in enumerate(data['middleAccuracyKSplits']):
        for itr, accuracy in enumerate(array):
            crossval_arr.append({'strategy': 'middle', 'iteration': itr, 'run': ind, 'accuracy': accuracy})

    for ind, array in enumerate(data['correctionAccuracyKSplits']):
        for itr, accuracy in enumerate(array):
            crossval_arr.append({'strategy': 'correction', 'iteration': itr, 'run': ind, 'accuracy': accuracy})

    for ind, array in enumerate(data['labelAccuracyKSplits']):
        for itr, accuracy in enumerate(array):
            crossval_arr.append({'strategy': 'label', 'iteration': itr, 'run': ind, 'accuracy': accuracy})

    # ------------------------------------------------------------------------------------------------------------------

    for itr, accuracy in enumerate(data['randomAccuracyAverage']):
        crossvalavg_arr.append({'strategy': 'random', 'iteration': itr, 'accuracy': accuracy})

    for itr, accuracy in enumerate(data['uncertaintyAccuracyAverage']):
        crossvalavg_arr.append({'strategy': 'uncertainty', 'iteration': itr, 'accuracy': accuracy})

    for itr, accuracy in enumerate(data['middleAccuracyAverage']):
        crossvalavg_arr.append({'strategy': 'middle', 'iteration': itr, 'accuracy': accuracy})

    for itr, accuracy in enumerate(data['correctionAccuracyAverage']):
        crossvalavg_arr.append({'strategy': 'correction', 'iteration': itr, 'accuracy': accuracy})

    for itr, accuracy in enumerate(data['labelAccuracyAverage']):
        crossvalavg_arr.append({'strategy': 'label', 'iteration': itr, 'accuracy': accuracy})

xvavg_df = pd.DataFrame(data=crossvalavg_arr)

# create arrays for ttest
vis_sel_precision = []
lst_sel_precision = []

vis_sel_recall = []
lst_sel_recall = []

vis_cls_accuracy = []
lst_cls_accuracy = []

for user, user_data in result_dict.items():
    for task in user_data['short']:
        if task['task'] == 's':
            if task['layout'] == 'v':
                vis_sel_precision.append(task['selectedPrecision'])
                vis_sel_recall.append(task['selectedRecall'])
            elif task['layout'] == 'l':
                lst_sel_precision.append(task['selectedPrecision'])
                lst_sel_recall.append(task['selectedRecall'])
        elif task['task'] == 'c':
            if task['layout'] == 'v':
                vis_cls_accuracy.append(task['accuracy'])
            elif task['layout'] == 'l':
                lst_cls_accuracy.append(task['accuracy'])

# t-test sample based -> use scipy.stats.ttest_1samp
# create t-test for task s (precision & recall)
ttest_prec = ttest_ind(vis_sel_precision, lst_sel_precision)
ttest_rec = ttest_ind(vis_sel_recall, lst_sel_recall)

# create t-test for task c (accuracy)
# do accuracy after 5 min
ttest_acc = ttest_ind(vis_cls_accuracy, lst_cls_accuracy)

# TODO
# do accuracy after same number of labels
# find lowest count of labels
# retrieve accuracy from raw

sel_arr = []
cls_arr = []
all_arr = []

for user, user_data in result_dict.items():
    for task in user_data['short']:
        all_arr.append(task)
        if task['task'] == 's':
            sel_arr.append(task)
        elif task['task'] == 'c':
            cls_arr.append(task)
    for data in user_data['raw']:
        experiment = json.loads(data['experiment'])
        # task 2 = s, task 3 = c
        if experiment['task'] == 'task3':
            itr = 0
            for action in experiment['progressiveMeasures']:
                if action['modelOperation'] == 'update':
                    crossval_arr.append({'strategy': 'active ' + experiment['view'], 'iteration': itr, 'accuracy': action['accuracy']})
                    itr += 1

xv_df = pd.DataFrame(data=crossval_arr)

cls_df = pd.DataFrame(data=cls_arr)
sel_df = pd.DataFrame(data=sel_arr)

# plot results of study

ax = sns.catplot(x="layout", y="accuracy", kind="box", data=cls_df)
ax.fig.suptitle('Accuracy in Classification Task')
# sns.catplot(x="layout", y="accuracy", data=cls_df, ax=ax, palette=sns.color_palette(['gray']))

# sns.catplot(x="layout", y="selectedPrecision", data=sel_df)
ax = sns.catplot(x="layout", y="selectedPrecision", kind="box", data=sel_df)
ax.fig.suptitle('Precision in Selection Task')

# sns.catplot(x="layout", y="selectedRecall", data=sel_df)
ax = sns.catplot(x="layout", y="selectedRecall", kind="box", data=sel_df)
ax.fig.suptitle('Recall in Selection Task')

# add users to classification chart (accuracy)
# retrieve accuracy values from raw data
# add them to the plot

a4_dims = (18, 12)
fig, ax = plt.subplots(figsize=a4_dims)
plt.title('Accuracy Comparison between Strategies')
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color=(0.1, 0.1, 0.1, 0.2))
plt.tick_params(which='both',  # Options for both major and minor ticks
                top='off',  # turn off top ticks
                left='off',  # turn off left ticks
                right='off',  # turn off right ticks
                bottom='off')
sns.lineplot(x="iteration", y="accuracy", hue="strategy", data=xv_df, ax=ax)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig.show

a4_dims = (18, 12)
fig, ax = plt.subplots(figsize=a4_dims)
plt.title('Accuracy Averages Comparison between Strategies')
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color=(0.1, 0.1, 0.1, 0.2))
plt.tick_params(which='both',  # Options for both major and minor ticks
                top='off',  # turn off top ticks
                left='off',  # turn off left ticks
                right='off',  # turn off right ticks
                bottom='off')
sns.lineplot(x="iteration", y="accuracy", hue="strategy", data=xvavg_df, ax=ax)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig.show

plt.show()
