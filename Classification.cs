using System;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Data;
using System.Collections.Generic;

namespace MLDemo
{
    ///<summary> 
    ///Author: John Ferguson
    ///Date: 5th June 2019
    ///======================================================================================================================================
    ///MLClassificationModel is an object that can be used to perform Binary Classification Supervised Machine Learning against a Data Set.
    /// Sequence of Events:
    /// 1 - Load the Data
    /// 2 - Train the Model
    /// 3 - Save the Model
    /// 4 - Evaluate the Model (Optional)
    /// 5 - Load the Model (at some point in the future)
    /// 6 - Predict against the Model
    /// This particular implementation uses the LBFGS Logistical Regression Algorithm
    /// 
    /// The Model is designed to perform Sentinment Analysis against a very large data set of pre-labelled Tweets exported from Twitter.
    /// The data set is split into Postitive (1) and Negative (0) Tweets. The data set contains a series of columns that are not relevant for training.
    /// The objective is to predict the Positive or Negative sentiment of a new value.
    ///</summary>    
    public class MLClassificationModel
    {
        private readonly MLContext objContext = new MLContext();

        private IDataView DataView { get; set; }
        private ITransformer Model { get; set; }
        private TrainTestData SplitDataView { get; set; }

        ///<summary>
        ///Create a training & test data set given a valid path of a data file. The data is expected to be comma-delimited and must be in the format as laid out in the ClassificationInputSchema class.
        ///</summary>
        ///<param name="filePath">The local file-system path of the data set to import</param>
        ///<param name="testPercentage">The proportion of the dataset to be used for Testing (as opposed to Training)</param>        
        public void LoadData(string filePath, double testPercentage = 0.2)
        {
            // Build a data view with the contents of the file - the file schema must match the schema laid out in ClassificationInputSchema
            DataView = objContext.Data.LoadFromTextFile<ClassificationInputSchema>(filePath, separatorChar:',',hasHeader: false);

            // Split the data into two data sets: Train & Test. Split percentage is based on a parameter provided by the calling program
            SplitDataView = objContext.Data.TrainTestSplit(DataView, testFraction: testPercentage);
        }

        ///<summary>
        ///The Enumerable Data View is a version of the Data Set which can be iterated. Used for demo purposes
        ///<param name="isTrain"/>Determines if the Training Data Set should be used. If False, the Test Set is returned instead</param>
        ///</summary>  
        public IEnumerable<ClassificationInputSchema> enumerableDataView(bool isTrain = true)
        {
            if (isTrain)
            {
                return objContext.Data.CreateEnumerable<ClassificationInputSchema>(SplitDataView.TrainSet , reuseRowObject: true);
            }
            else
            {
                return objContext.Data.CreateEnumerable<ClassificationInputSchema>(SplitDataView.TestSet, reuseRowObject: true);
            }
        }

        ///<summary>
        ///Use a training data set to produce a Model by featurising the text in the data set
        ///</summary>
        public void Train()
        {
            /*
             * A pipeline is a collection of execution steps used to perform ML training. In this example, there are only two steps:
             ** Featurize: breaks the Text column into a collection of features (N-grams) 
             ** Trainer: specifies the ML training alogrithm to use to produce a Model
             * The pipeline is based on 'lazy execution' - these actions are not performed until the Fit method is called
             */
            var pipeline = objContext.Transforms.Text.FeaturizeText("Features", "Text")                
                .Append(objContext.BinaryClassification.Trainers.LbfgsLogisticRegression (labelColumnName: "Sentiment", featureColumnName: "Features"));

            // Execute the pipeline against the Training data, and produce a Model
            Model = pipeline.Fit(SplitDataView.TrainSet);
        }

        ///<summary>
        ///Output the completed model to the file system.
        ///</summary>
        ///<param name="filepath">The local file-system path to store the Model. Should include a .zip file extension</param>
        public void Save(string filepath)
        {
            objContext.Model.Save(Model, SplitDataView.TrainSet.Schema, filepath);
        }

        ///<summary>
        ///Load a previously generated Model from the file system. The data file used to generate the Model can also be imported, if the aim is to evaluate or retrain the model
        ///</summary>
        ///<param name="datafilepath">The local file-system path of the Data Set that was used to generate the Model. Only required if you want to perform an Evaluation</param>
        ///<param name="modelfilepath">The local file-system path of the Model to load</param>
        public void Load(string modelfilepath, string datafilepath = null)
        {
            DataViewSchema modelSchema;
            Model = objContext.Model.Load(modelfilepath, out modelSchema);
            if (datafilepath != null)
            {
                LoadData(datafilepath);
            } 
        }

        ///<summary>
        ///Produce a set of metrics of a Model by running a set of Test data against it.
        ///</summary>
        public BinaryClassificationMetrics Evaluate()
        {
            var testData = Model.Transform(SplitDataView.TestSet);
            BinaryClassificationMetrics bcmMetrics = objContext.BinaryClassification.Evaluate(data: testData, labelColumnName: "Sentiment", predictedLabelColumnName: "PredictedLabel", scoreColumnName: "Score", probabilityColumnName: "Probability");
            return bcmMetrics;
        }

        ///<summary>
        ///Use a loaded Model to perform a prediction against a new text value.
        ///</summary>
        ///<param name="input">A string of text representing a Tweet that you want to determine the Sentiment of</param>
        public ClassificationOutputSchema Predict(string input)
        {
            
            // Build a prediction Engine by loading the Model
            var predictionEngine = objContext.Model.CreatePredictionEngine<ClassificationInputSchema, ClassificationOutputSchema>(Model);

            // Create a new data sampler using the text input as a parameter
            ClassificationInputSchema sample = new ClassificationInputSchema
            {                
                text = input
            };

            // Generate a prediction against the sampler
            var result = predictionEngine.Predict(sample);

            return result;
        }

    }

    ///<Summary>
    ///The input schema describes the structure of the inbound data set that we want to use for ML
    ///</Summary>
    public class ClassificationInputSchema
    {
        [LoadColumn(0), ColumnName("Sentiment")]
        public bool sentiment;

        [LoadColumn(1), ColumnName("ID")]
        public Int32 id;

        [LoadColumn(2), ColumnName("Date")]
        public string date;

        [LoadColumn(3), ColumnName("Flag")]
        public string flag;

        [LoadColumn(4), ColumnName("User")]
        public string user;

        [LoadColumn(5), ColumnName("Text")]
        public string text;

    }

    ///<Summary>
    ///The output schema describes the structure of the Prediction result that will be generated by the Predict method. The structure is dictated by the alogrithm used.
    ///</Summary>
    public class ClassificationOutputSchema
    {

        [ColumnName("PredictedLabel")]
        public bool predictedLabel { get; set; }

        [ColumnName("Probability")]
        public float probability { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }

        // Friendly Label is used for demo purposes to turn the Boolean result into a meaningful result
        public string friendlyLabel()
        {
            if (predictedLabel)
            {
                return "Positive";
            }
            else
            {
                return "Negative";
            }
        }

        // Friendly Probability is used for demo purposes to turn the decimal probability into a string
        public string friendlyProbability()
        {            
            return $"{probability:##.#%}";
        }

        // Confidence is used for demo purposes and provides a text description of the probability
        public string Confidence()
        {
            string output = "Low";

            if (probability < 0.4)
            {
                output = "High";
            }

            if (probability < 0.2)
            {
                output = "Very High";    
            }

            if (probability > 0.6)
            {
                output = "High";
            }

            if (probability > 0.8)
            {
                output = "Very High";
            }

            return output;

        }
    }

}

