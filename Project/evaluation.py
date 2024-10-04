import re

'''
Evaluation Scenario I: Input-> friends folder and Real Image of me (Subhan)

{'True Positive': 28, 'False Positive': 22, 'True Negative': 0, 'False Negative': 0, 'Accuracy': 0.56, 'Precision': 0.56, 'Recall': 1.0, 'F1 Score': 0.717948717948718, 'Specificity': 0.0}
'''


def extract_numbers(text):
    # Use regular expression to find all numbers in the string
    numbers = re.findall(r'\d+', text)
    # Convert the extracted numbers from strings to integers
    numbers = [int(num) for num in numbers]
    return numbers


original = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15, 19, 21,
            29, 33, 34, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50]

predicted_arr = ['./input/friends/28.jpg', './input/friends/29.jpg', './input/friends/3.jpg', './input/friends/30.jpg', './input/friends/31.jpg', './input/friends/32.jpg', './input/friends/33.jpg', './input/friends/34.jpg', './input/friends/35.jpg', './input/friends/36.jpg', './input/friends/37.jpg', './input/friends/38.jpg', './input/friends/39.jpg', './input/friends/4.jpg', './input/friends/40.jpg', './input/friends/41.jpg', './input/friends/42.jpg', './input/friends/43.jpg', './input/friends/44.jpg', './input/friends/45.jpg', './input/friends/46.jpg', './input/friends/47.jpg', './input/friends/48.jpg', './input/friends/49.jpg', './input/friends/5.jpg',
                 './input/friends/50.jpg', './input/friends/6.jpg', './input/friends/7.jpg', './input/friends/8.jpg', './input/friends/9.jpg', './input/friends/1.jpg', './input/friends/10.jpg', './input/friends/11.jpg', './input/friends/12.jpg', './input/friends/13.jpg', './input/friends/14.jpg', './input/friends/15.jpg', './input/friends/16.jpg', './input/friends/17.jpg', './input/friends/18.jpg', './input/friends/19.jpg', './input/friends/2.jpg', './input/friends/20.jpg', './input/friends/21.jpg', './input/friends/22.jpg', './input/friends/23.jpg', './input/friends/24.jpg', './input/friends/25.jpg', './input/friends/26.jpg', './input/friends/27.jpg']

extracted_nums = []

for img in predicted_arr:
    extracted_nums.append(extract_numbers(img)[0])


truePositive = 0
trueNegative = 0
falsePositive = 0
print(len(extracted_nums))
for i in range(1, 51):
    if i not in original and i not in extracted_nums:
        trueNegative += 1

for i in extracted_nums:
    if i not in original:
        falsePositive += 1
    else:
        original.remove(i)
        truePositive += 1


falseNegative = len(original)


def calculate_metrics(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0

    # Calculate specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Create a dictionary to store the metrics
    metrics = {
        'True Positive': tp,
        'False Positive': fp,
        'True Negative': tn,
        'False Negative': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Specificity': specificity
    }

    return metrics


print(calculate_metrics(truePositive, falsePositive, trueNegative, falseNegative))

'''
Evaluation Scenario II: Input-> friends folder and Image Segregation

{'True Positive': 78, 'False Positive': 23, 'True Negative': 0, 'False Negative': 3, 'Accuracy': 0.75, 'Precision': 0.7722772277227723, 'Recall': 0.9629629629629629, 'F1 Score': 0.8571428571428571, 'Specificity': 0.0}
'''

trueNegative = 0
truePositive = 78
falseNegative = 3
falsePositive = 23
print(calculate_metrics(truePositive, falsePositive, trueNegative, falseNegative))

'''
Evaluation Scenario III: Input-> fam2a folder and Image Segregation
{'True Positive': 85, 'False Positive': 4, 'True Negative': 6, 'False Negative': 14, 'Accuracy': 0.8348623853211009, 'Precision': 0.9550561797752809, 'Recall': 0.8585858585858586, 'F1 Score': 0.9042553191489363, 'Specificity': 0.6}
'''
truePositive = 85
trueNegative = 6
falsePositive = 4
falseNegative = 14

print(calculate_metrics(truePositive, falsePositive, trueNegative, falseNegative))
