from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd 
import argparse
from pathlib import Path
from matplotlib import pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument('--input', help='Input file path', required=True)
argparser.add_argument('--output', help='Output directory path', required=True)
argparser.add_argument('--model', help='The type of model to use', required=True)

args = argparser.parse_args()

input_file = Path(args.input)
output_dir = Path(args.output)

model = args.model

data = pd.read_csv(input_file)

# drop any columns with more than 50% NaNs
data = data.dropna(thresh=len(data) // 2, axis=1)
# drop any rows with NaNs
data = data.fillna(0)

Qs = data['Q'].unique().tolist()
Qs.sort()
lowest_Q = Qs.pop(0)



data['scaled'] = (data['Q'] == lowest_Q).astype(int)



feature_filters = {
    'Fixation Times': r'm\d_fixation',
    'Site Frequency Spectrum': r'm\d_sfs',
    'Fixation Probabilities': r'fixation_prob_\d+',
    'LD': r'ld_\d+',
}

if model == 'lr':
    clf = LogisticRegression()
    title = 'Accuracy of Logistic Regression Classifier'
    file_name = 'lr_accuracy'
elif model == 'rf':
    title = 'Accuracy of Random Forest Classifier'
    clf = RandomForestClassifier(n_estimators=300, max_depth=10)
    file_name = 'rf_accuracy'


score_dict = {}
line_types = ['-', '--', '-.', ':']
marker_types = ['o', 'v', 's', 'D', 'P', 'X', 'd', 'p', 'h', 'H', '*', '1', '2', '3', '4', '8', 'x', '+', 'D', '|', '_']
accuracy_df = pd.DataFrame(columns=['Q', 'Feature', 'Accuracy'])

for i, (feature_type, feature_filter) in enumerate(feature_filters.items()):
    feature_df = data.filter(regex=feature_filter)
    scores = []

    for Q in Qs:
        x = feature_df[(data['Q'] == Q) | (data['Q'] == lowest_Q)]
        scaler = StandardScaler()
        # if model == 'lr':
        #     x = scaler.fit_transform(x)
        x = scaler.fit_transform(x)
        y = data['scaled'][(data['Q'] == Q) | (data['Q'] == lowest_Q)]
        # shuffle and split training and test sets
        x, y = shuffle(x, y)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        # train model
        clf.fit(X_train, y_train)
        # test model
        score = clf.score(X_test, y_test)
        scores.append(score)
        accuracy_df = pd.concat([accuracy_df, pd.DataFrame({'Q': [Q], 'Feature': [feature_type], 'Accuracy': [score]})])
    
    linetype = line_types[i % len(line_types)]
    marker = marker_types[i % len(marker_types)]
    # plot lines with triangles at each point
    plt.plot(Qs, scores, label=feature_type, linestyle=linetype, marker=marker, alpha=0.50)
    # make the x axis ticks the Q values
    plt.xticks(Qs, Qs)

plt.legend()
plt.xlabel('Q')
plt.ylabel('Accuracy')
plt.title(title)

plt.savefig(output_dir / f'graphs/{file_name}.svg', bbox_inches='tight')
accuracy_df.to_csv(output_dir / f'summary_stats/{file_name}.csv', index=False)

