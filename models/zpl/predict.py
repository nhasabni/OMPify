import torch
from statistics import mean

def predict(model, test_name, test_loader, threshold):
  with torch.no_grad():
    false_positive = 0
    true_positive = 0
    false_negative = 0
    true_negative = 0
    
    for step, (test_code, omp_pragma_exists) in enumerate(test_loader):
      # Output of Autoencoder
      reconstructed = model(test_code)
      
      # Calculating squared errors for reconstruction
      # shape is [32 x 256]
      squared_errors = torch.nn.MSELoss(reduction='none')(reconstructed, test_code).numpy()

      # avg squared error across every test code
      avg_squared_errors = [mean(x) for x in squared_errors]

      for i in range(len(avg_squared_errors)):
        # If we can reconstruct the input that had omp_pragma, then it is correctly
        # classified as a part of the distribution.
        if (avg_squared_errors[i] < threshold and omp_pragma_exists[i].numpy() == 1):
          true_positive += 1
        # or if we cannot reconstruct out-of-distribution input, then it is also
        # correct classification.
        elif (avg_squared_errors[i] > threshold and omp_pragma_exists[i].numpy() == 0):
          true_negative += 1
        elif (avg_squared_errors[i] < threshold and omp_pragma_exists[i].numpy() == 0):
          # otherwise, it is wrong classification.
          false_positive += 1
        else:
          false_negative += 1
        #print(avg_squared_errors[i], omp_pragma_exists[i],
        #			"TP:", true_positive, "TN:", true_negative,
        #			"FP:", false_positive, "FN:", false_negative)

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    f1_score = (2 * (precision * recall)) / (precision + recall)
    
    print("Results for dataset:", test_name)
    print("accuracy: {:2.2%}, recall: {:2.2%}, precision: {:2.2%}, f1: {:2.2%}".format(accuracy, recall, precision, f1_score))
