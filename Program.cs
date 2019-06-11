using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Data;
using System.Globalization;

namespace MLDemo
{
    enum TaskType
    {
        CLASSIFICATION = 1,
        CLUSTERING = 2,
        REGRESSION = 3,
        RECOMMENDATION = 4,
        MULTI_CLASSIFICATION = 5,
        ANOMALY_DETECTION = 6
    }

    ///<summary> 
    ///Author: John Ferguson
    ///Date: 5th June 2019
    ///======================================================================================================================================
    /// The Program provides console interface to the six Machine Learning demos
    /// 
    static class Program
    {
        static void Main(string[] args)
        {
                                            
            bool blnNewModel = false;
            TaskType type = TaskType.CLASSIFICATION;
           
            string filepath;
            string modelpath;
            string resultpath;
            bool xEsc;
            string input;

            #region Establish Demo Type

            // Establish which Machine Learning demo the user wants to perform
            Console.WriteLine("What type of Machine Learning do you want to do?");
            Console.WriteLine("1 = Binary Classification");
            Console.WriteLine("2 = Clustering");
            Console.WriteLine("3 = Regression (e.g. Prediction)");
            Console.WriteLine("4 = Recommendation");
            Console.WriteLine("5 = Multi-class Classification");
            Console.WriteLine("6 = Anomaly Detection");
            input = Console.ReadLine();
            switch(Convert.ToInt16(input))
            {
                case 1:
                    type = TaskType.CLASSIFICATION;
                    break;

                case 2:
                    type = TaskType.CLUSTERING;
                    break;

                case 3:
                    type = TaskType.REGRESSION;
                    break;

                case 4:
                    type = TaskType.RECOMMENDATION;
                    break;

                case 5:
                    type = TaskType.MULTI_CLASSIFICATION;
                    break;

                case 6:
                    type = TaskType.ANOMALY_DETECTION;
                    break;

                default:
                    Console.WriteLine("Unrecognised Choice");
                    Environment.Exit(0);
                    break;
            }

            // We don't do model re-training in the Anomaly Detection demo, so don't ask the question
            if (type != TaskType.ANOMALY_DETECTION)
            {
                // Establish if the user wants to retrain the model in the demo, or just reload a previously defined model (faster)
                Console.WriteLine("Do you want to retrain the model? (y/n) - (Choose 'y' if you have not run the example before)");
                input = Console.ReadLine();
                blnNewModel = (input == "y");

                Console.WriteLine("");
                Console.WriteLine("==========================");

            }

            #endregion

            switch (type)
            {
                
                case TaskType.CLASSIFICATION:

                    /*
                     * Classification is a type of Supervised Machine Learning in which patterns are extracted from a series of 'Features' and used to group rows of data
                     * into different classes based on a provided Label. The Model can then be used to predict which Label / Category a new item should belong to, given its features.
                     * There are two types of Classification: Binary & Multi-class. This demo is an example of a Binary Classifcation, in which rows are deemed either Positive or Negative.                      
                     * In this example, we will determine whether a sentence has either a postive or negative 'sentiment'.
                     */
                    #region Classification
                    // Create an instance of the Classfication ML class
                    MLClassificationModel mlMClassification = new MLClassificationModel();

                    filepath = BuildPath("twitterdata.csv");
                    modelpath = BuildPath("model.zip");

                    // If the user wants to retrain / regenerate the Model...
                    if (blnNewModel)
                    {
                        Console.WriteLine("Expressed as a decimal, what percentage of the data set do you want to reserve for testing purposes? (recommended value is '0.2' - e.g. 20%)");
                        input = Console.ReadLine();

                        Console.WriteLine("Please be patient; this may take a few minutes.");

                        // Load, Train and Save the Model
                        mlMClassification.LoadData(filepath, float.Parse(input, CultureInfo.InvariantCulture.NumberFormat));
                        mlMClassification.Train();
                        mlMClassification.Save(modelpath);

                        // Score the model using the Test data, and generate a Metrics bundle
                        BinaryClassificationMetrics metrics = mlMClassification.Evaluate();

                        // Output the metrics to the screen
                        Console.WriteLine($"Area Under the ROC Curve (closer to 1 the better):  + {metrics.AreaUnderRocCurve:#.##}");
                        Console.WriteLine($"Accuracy (closer to 1 the better):  + {metrics.Accuracy:#.##}");
                        Console.WriteLine($"F1 Score (closer to 1 the better, ideally should be close to Accuracy):  + {metrics.F1Score:#.##}");
                        Console.WriteLine("Confusion Matrix: (You're looking for a balance between Positive & negative predictions, and a relatively low number of False Positives & False Negatives)");
                        Console.WriteLine("");
                        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
                    }
                    else
                    {
                        // The user does not want to retrain the model, therefore load it from file
                        mlMClassification.Load(modelpath, filepath);               
                    }

                    xEsc = true;

                    do
                    {
                        Console.WriteLine("");
                        Console.WriteLine("==========================");
                        Console.WriteLine("Type a sentence: ");
                        input = Console.ReadLine();

                        // Generate and output a prediction given the input text entered by the user
                        ClassificationOutputSchema pred = mlMClassification.Predict(input);
                        Console.WriteLine("Predicted sentiment of the sentence: " + pred.friendlyLabel());
                        Console.WriteLine("Confidence: " + pred.Confidence());
                        Console.WriteLine("Probability: " + pred.friendlyProbability() + " (0 = Max. Negative, 50 = Neutral, 100 = Max. Positive)");
                        Console.WriteLine("");

                    } while (xEsc);

                    break;
                #endregion

                case TaskType.CLUSTERING:

                    /*
                     * Clustering is a type of Unsupervised Machine Learning in which rows are grouped into one of a number of 'Clusters' based on patterns in the data. The difference
                     * with Classification is that the data is not pre-labelled; the machine is not told in advance which Cluster / Category the row belongs to. The clustering is done
                     * organically, purely based on patterns within the Features.
                     * In this example, we will cluster a set of flowers into one of three species.
                     */

                    #region Clustering
                    // Create an instance of the Clustering ML class
                    MLClusteringModel mlCluster = new MLClusteringModel();

                    filepath = BuildPath("iris.csv");
                    modelpath = BuildPath("clustermodel.zip");
                    resultpath = BuildPath("clusterresult.csv");

                    // If the user wants to retrain / regenerate the Model...
                    if (blnNewModel)
                    {
                        // Load, Train and Save the Model
                        mlCluster.LoadData(filepath);
                        mlCluster.Train();
                        mlCluster.Save(modelpath);

                        // Create an output file to write the results to
                        using (System.IO.StreamWriter file =
                            new System.IO.StreamWriter(resultpath, false))
                        {
                            // Iterate through all items in the original Iris Flower data set
                            IEnumerable<ClusteringInputSchema> irisData = mlCluster.enumerableDataView();
                            foreach (ClusteringInputSchema row in irisData)
                            {
                                // Predict which cluster each flower belongs to, and write this line out to the Output file
                                ClusteringOutputSchema result = mlCluster.Predict(row);
                                file.WriteLine(row.flattenedText() + ", " + result.PredictedCluster);
                            }

                        }

                        Console.WriteLine("Output file written to: " + resultpath);

                    }
                    else
                    {
                        // The user does not want to retrain the model, therefore load it from file
                        mlCluster.Load(modelpath);                  
                    }


                    string sepalLength, sepalWidth, petalLength, petalWidth;
                    xEsc = true;

                    do
                    {
                        // Grab the input variables from the user in order to make a prediction
                        Console.WriteLine("");
                        Console.WriteLine("==========================");
                        Console.WriteLine("Enter Petal Length: ");
                        petalLength = Console.ReadLine();
                        Console.WriteLine("Enter Petal Width: ");
                        petalWidth = Console.ReadLine();
                        Console.WriteLine("Enter Sepal Length: ");
                        sepalLength = Console.ReadLine();
                        Console.WriteLine("Enter Sepal Width: ");
                        sepalWidth = Console.ReadLine();
                        
                        //Build an input object given the variables provided
                        ClusteringInputSchema sample = new ClusteringInputSchema()
                        {
                            petalLength = Convert.ToSingle(petalLength),
                            petalWidth = Convert.ToSingle(petalWidth),
                            sepalLength = Convert.ToSingle(sepalLength),
                            sepalWidth = Convert.ToSingle(sepalWidth)
                        };

                        //Use the input object to make a prediction
                        ClusteringOutputSchema prediction = mlCluster.Predict(sample);

                        Console.WriteLine("NOTE: Distance represents how close to the 'centre' of each cluster this value is.");
                        Console.WriteLine("Predicted Cluster: " + prediction.PredictedCluster + prediction.flattenedDistances());
                        Console.WriteLine("");

                    } while (xEsc);
                    
                    break;
                #endregion

                case TaskType.REGRESSION:

                    /*
                     * Regression is a type of Supervised Machine Learning which allows a value to be predicted based on the value within historical examples which have similar associated variables.
                     * The associated variables can one or many fields, that may or may not have any relationship with the value itself.
                     * In this example, we will predict a taxi fare, given a set of input variables (distance, time, passengers, etc) by looking at thousands of historical taxi journeys.
                     * 
                     */

                    #region Regression
                    // Create an instance of the Regression ML class
                    MLRegressionModel mlRegression = new MLRegressionModel();

                    filepath = BuildPath("taxi.csv");
                    modelpath = BuildPath("regressionmodel.zip");

                    // If the user wants to retrain / regenerate the Model...
                    if (blnNewModel)
                    {
                        Console.WriteLine("Expressed as a decimal, what percentage of the data set do you want to reserve for testing purposes? (recommended value is '0.2' - e.g. 20%)");
                        input = Console.ReadLine();

                        // Load, Train and Save the Model
                        mlRegression.LoadData(filepath, float.Parse(input, CultureInfo.InvariantCulture.NumberFormat));
                        mlRegression.Train();
                        mlRegression.Save(modelpath);

                        // Score the model using the Test data, and generate a Metrics bundle
                        RegressionMetrics metrics = mlRegression.Evaluate();

                        // Output the metrics to the screen
                        Console.WriteLine($"RSquared:  + {metrics.RSquared:#.##}");
                        Console.WriteLine($"Root Mean Squared:  + {metrics.RootMeanSquaredError:#.##}");
                        Console.WriteLine($"Mean Absolute Error:  + {metrics.MeanAbsoluteError:#.##}");
                        Console.WriteLine($"Mean Square Error:  + {metrics.MeanSquaredError:#.##}");

                    }
                    else
                    {
                        // The user does not want to retrain the model, therefore load it from file
                        mlRegression.Load(modelpath);
                    }


                    string vendorId, rateCode, passengerCount, tripTime, tripDistance, paymentType;
                    xEsc = true;

                    do
                    {
                        // Grab the input variables from the user in order to make a prediction
                        Console.WriteLine("");
                        Console.WriteLine("==========================");
                        Console.WriteLine("Enter Vendor ID (Suggested VTS): ");
                        vendorId = Console.ReadLine();
                        Console.WriteLine("Enter Rate Code (Suggested 1): ");
                        rateCode = Console.ReadLine();
                        Console.WriteLine("Enter Passenger Count: ");
                        passengerCount = Console.ReadLine();
                        Console.WriteLine("Enter Trip Time in Seconds (Suggested 1000): ");
                        tripTime = Console.ReadLine();
                        Console.WriteLine("Enter Trip Distance in Miles (Suggested 3.5): ");
                        tripDistance = Console.ReadLine();
                        Console.WriteLine("Enter Payment Type (Suggested CRD): ");
                        paymentType = Console.ReadLine();

                        //Build an input object given the variables provided
                        RegressionInputSchema regressionSample = new RegressionInputSchema()
                        {
                            vendorId = vendorId,
                            rateCode = float.Parse(rateCode, CultureInfo.InvariantCulture.NumberFormat),
                            passengerCount = float.Parse(passengerCount, CultureInfo.InvariantCulture.NumberFormat),
                            tripTime = float.Parse(tripTime, CultureInfo.InvariantCulture.NumberFormat),
                            tripDistance = float.Parse(tripDistance, CultureInfo.InvariantCulture.NumberFormat),
                            paymentType = paymentType
                        };

                        //Use the input object to make a prediction
                        RegressionOutputSchema prediction = mlRegression.Predict(regressionSample);

                        Console.WriteLine("Predicted Fare: " + prediction.cleanFare());
                        Console.WriteLine("");

                    } while (xEsc);

                    break;
                #endregion


                case TaskType.RECOMMENDATION:

                    /*
                     * Recommendation is a type of Regression-based Supervised Machine Learning which predicts the relationship between and X-type object & Y-type object based on the relationships
                     * that already exist between X and other Ys, and Y and other Xs. 
                     * In this example we will generate a list of movie recommendations for a user based on their (and other user's) ratings of other movies.
                     * 
                     */

                    #region Recommendation

                    MLRecommendationModel mLRecommendation = new MLRecommendationModel();

                    filepath = BuildPath("movieratings.csv");
                    string refpath = BuildPath("movieref.csv");
                    modelpath = BuildPath("recommendationnmodel.zip");

                    // If the user wants to retrain / regenerate the Model...
                    if (blnNewModel)
                    {
                        Console.WriteLine("Expressed as a decimal, what percentage of the data set do you want to reserve for testing purposes? (recommended value is '0.2' - e.g. 20%)");
                        input = Console.ReadLine();

                        // Load, Train and Save the Model
                        mLRecommendation.LoadData(filepath, refpath, float.Parse(input, CultureInfo.InvariantCulture.NumberFormat));

                        Console.WriteLine("How many learning iterations do you want to do? Higher values will yield more accurate models, but will take longer (recommended value is 20)");
                        input = Console.ReadLine();
                        mLRecommendation.Train(Convert.ToInt16(input));

                        mLRecommendation.Save(modelpath);

                        // Score the model using the Test data, and generate a Metrics bundle
                        RegressionMetrics metrics = mLRecommendation.Evaluate();

                        // Output the metrics to the screen
                        Console.WriteLine($"RSquared:  + {metrics.RSquared:#.##}");
                        Console.WriteLine($"Root Mean Squared:  + {metrics.RootMeanSquaredError:#.##}");
                        Console.WriteLine($"Mean Absolute Error:  + {metrics.MeanAbsoluteError:#.##}");
                        Console.WriteLine($"Mean Square Error:  + {metrics.MeanSquaredError:#.##}");

                    }
                    else
                    {
                        // The user does not want to retrain the model, therefore load it from file
                        mLRecommendation.Load(modelpath, refpath);
                    }

                    xEsc = true;
                    string userId, recCount;

                    do
                    {
                        
                        // Grab the input variables from the user in order to make a prediction
                        Console.WriteLine("");
                        Console.WriteLine("==========================");
                        Console.WriteLine("Select a User ID (Suggested 6): ");
                        userId = Console.ReadLine();
                        Console.WriteLine("How many recommendations do you want to see? (Suggested 5): ");
                        recCount = Console.ReadLine();
                        Console.WriteLine("Generating Recommendations...");
                        List<MovieRecommendation> recommendations = mLRecommendation.GetRecommendations(Convert.ToInt32(userId), Convert.ToInt32(recCount));
                        int iter = 1;
                        foreach (MovieRecommendation r in recommendations)
                        {
                            Console.WriteLine("Rank: #" + iter.ToString() + ": " + r.title + " (" + r.movieId + "), Score: " + r.score.ToString());
                            iter++;
                        }
                        Console.WriteLine("");

                    } while (xEsc);

                    break;

                #endregion

                case TaskType.MULTI_CLASSIFICATION:

                    /*
                     * Multi-Class Classification is similar to Classification, except rather than grouping items into Binary (true / false) groups, it can support multiple, labelled groups.
                     * In this example we will predict the support area of a system issue by looking at historical GitHub tickets and deriving patterns from the title and issue text.
                     */

                    #region MultiClassification
                    // Create an instance of the MultiClassfication ML class
                    MLMultiClassificationModel mlMultiClassification = new MLMultiClassificationModel();

                    filepath = BuildPath("githubissues.tsv");
                    modelpath = BuildPath("multiclassmodel.zip");

                    // If the user wants to retrain / regenerate the Model...
                    if (blnNewModel)
                    {
                        Console.WriteLine("Expressed as a decimal, what percentage of the data set do you want to reserve for testing purposes? (recommended value is '0.2' - e.g. 20%)");
                        input = Console.ReadLine();

                        Console.WriteLine("Please be patient; this may take a few minutes.");

                        // Load, Train and Save the Model
                        mlMultiClassification.LoadData(filepath, float.Parse(input, CultureInfo.InvariantCulture.NumberFormat));
                        mlMultiClassification.Train();
                        mlMultiClassification.Save(modelpath);

                        // Score the model using the Test data, and generate a Metrics bundle
                        MulticlassClassificationMetrics metrics = mlMultiClassification.Evaluate();

                        // Output the metrics to the screen
                        Console.WriteLine($"Micro Accuracy (closer to 1 the better):  + {metrics.MicroAccuracy:#.##}");
                        Console.WriteLine($"Macro Accuracy (closer to 1 the better):  + {metrics.MacroAccuracy:#.##}");
                        Console.WriteLine($"Log Loss (closer to 0 the better):  + {metrics.LogLoss:#.##}");
                        Console.WriteLine($"Log Loss Reduction (closer to 0 the better):  + {metrics.LogLossReduction:#.##}");
                        Console.WriteLine("");
                        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
                    }
                    else
                    {
                        // The user does not want to retrain the model, therefore load it from file
                        mlMultiClassification.Load(modelpath, filepath);
                    }

                    string issueTitle, issueText;
                    xEsc = true;

                    do
                    {
                        Console.WriteLine("");
                        Console.WriteLine("==========================");
                        Console.WriteLine("Enter the Title of the new Support issue: ");
                        issueTitle = Console.ReadLine();
                        Console.WriteLine("Enter the Description of the Support issue: ");
                        issueText = Console.ReadLine();

                        MultiClassInputSchema issue = new MultiClassInputSchema();
                        issue.title = issueTitle;
                        issue.description  = issueText;
                        
                        // Generate and output a prediction given the input text entered by the user
                        MultiClassOutputSchema pred = mlMultiClassification.Predict(issue);
                        Console.WriteLine("Predicted Support Area: " + pred.area);
                        Console.WriteLine("");


                    } while (xEsc);

                    break;
                #endregion

                case TaskType.ANOMALY_DETECTION:

                    /*
                     * Anomaly Detection is a class of Machine Learning that detects deviation from the norm in a set of Time-Series data.
                     * In this example we will attempt to detect anomalous spikes in a set of sequential sales data.
                     */

                    #region AnomalyDetection
                    MLAnomalyDetectionModel mLAnomaly = new MLAnomalyDetectionModel();

                    Console.WriteLine("Which data set do you want to use? (1 = small, 2 = large)");
                    string dataset = Console.ReadLine();
                    string docSize = "";

                    // Determine if the user wants to use the small or large data set
                    if (Convert.ToInt16(dataset) == 1)
                    {
                        filepath = BuildPath("anomalydetection.csv");
                        docSize = "40";
                    }
                    else
                    {
                        filepath = BuildPath("anomalydetection2.csv");
                        docSize = "365";
                    }

                    resultpath = BuildPath("amomalydetectionresults.csv");

                    mLAnomaly.LoadData(filepath);

                    //Get user-defined variables
                    Console.WriteLine("What confidence (as a percentage) should be used before triggering a spike alert? (recommended 99)");
                    string confidence = Console.ReadLine();
                    Console.WriteLine("Approximately how many records are in the input file? (default = " + docSize + ")");
                    string recordCount = Console.ReadLine();
                    Console.WriteLine("What sensitivty factor should be used? (recommended 4)");
                    string sensitivity = Console.ReadLine();

                    //Build the Model
                    mLAnomaly.Train(Convert.ToInt16(confidence), Convert.ToInt16(sensitivity), Convert.ToInt16(recordCount));

                    string results = "";

                    // Create an output file to write the results to
                    using (System.IO.StreamWriter file =
                        new System.IO.StreamWriter(resultpath, false))
                    {

                       var predictions = mLAnomaly.DetectSpike();
                        foreach (AnomalyDetectionOutputSchema p in predictions)
                        {
                            // Vector 1 = Alert (0 = Normal, 1 = Spike Detected)
                            // Vector 2 = Score
                            // Vector 3 = P-Value (Calculated Probability)
                            results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";
                                
                            if (p.Prediction[0] == 1)
                            {
                                results += " <----------------- Spike detected";
                            }

                            Console.WriteLine(results);
                            file.WriteLine(p.Prediction[0].ToString() + "," + p.Prediction[1].ToString() + "," + p.Prediction[2].ToString());
                       
                        }

                        Console.WriteLine("Output file written to: " + resultpath);                        

                    }

                    break;
                #endregion AnomalyDetection

                default:
                    break;
            }
                              
                    
        }

        /// <summary>
        /// Generates the full local file-system path of a file, given the file name. Assumes the files are in the project's Data directory
        /// </summary>
        /// <param name="filename">The short name and extension of the file</param>
        /// <param name="datafolder">The name of the Data folder within the project directory</param>
        /// <returns></returns>
        public static string BuildPath(string filename, string datafolder = "Data")
        {
            string executingDirectory = Environment.CurrentDirectory;
            string binDirectory = Directory.GetParent(executingDirectory).Parent.FullName;
            string projectDirectory = Directory.GetParent(binDirectory).FullName;
            string dataFolderPath = Path.Join(projectDirectory, datafolder);
            string fullFilePath = Path.Join(dataFolderPath, filename);
            return fullFilePath;
        }

    }
}
