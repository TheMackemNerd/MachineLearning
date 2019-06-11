using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Data;

namespace MLDemo
{
    ///<summary> 
    ///Author: John Ferguson
    ///Date: 5th June 2019
    ///======================================================================================================================================
    ///MLRegressionModel is an object that can be used to perform Regression Supervised Machine Learning against a Data Set.
    /// Sequence of Events:
    /// 1 - Load the Data
    /// 2 - Train the Model
    /// 3 - Save the Model
    /// 4 - Load the Model (at some point in the future)
    /// 5 - Predict against the Model
    /// This particular implementation uses the Fast Tree Algorithm - NOTE: This had to be imported via Nuget independently of the Microsoft.ML library
    /// 
    /// The Model is designed to perform Linear Regression learning against a series of Taxi journeys. The aim is to identify patterns in the data
    /// and to predict the price of a new journey given a series of values.
    ///</summary>  
    public class MLRegressionModel
    {
        private readonly MLContext objContext = new MLContext();

        private IDataView dataView { get; set; }
        private ITransformer Model { get; set; }
        private TrainTestData SplitDataView { get; set; }

        ///<summary>
        ///Create a training & test data set given a valid path of a data file. The data is expected to be comma-delimited and must be in the format as laid out in the ClassificationInputSchema class.
        ///</summary>
        ///<param name="filePath">The local file-system path of the data set to import</param>
        ///<param name="testPercentage">The proportion of the dataset to be used for Testing (as opposed to Training)</param>    
        public void LoadData(string filePath, double testPercentage = 0.2)
        {
            // Build a data view with the contents of the file - the file schema must match the schema laid out in InputSchema
            dataView = objContext.Data.LoadFromTextFile<RegressionInputSchema>(filePath, separatorChar: ',', hasHeader: true);

            // Split the data into two data sets: Train & Test. Split percentage is based on a parameter provided by the calling program
            SplitDataView = objContext.Data.TrainTestSplit(dataView, testFraction: testPercentage);
        }

        ///<summary>
        ///The Enumerable Data View is a version of the Data Set which can be iterated. Used for demo purposes
        ///<param name="isTrain"/>Determines if the Training Data Set should be used. If False, the Test Set is returned instead</param>
        ///</summary>  
        public IEnumerable<RegressionInputSchema> enumerableDataView(bool isTrain = true)
        {
            if (isTrain)
            {
                return objContext.Data.CreateEnumerable<RegressionInputSchema>(SplitDataView.TrainSet, reuseRowObject: true);
            }
            else
            {
                return objContext.Data.CreateEnumerable<RegressionInputSchema>(SplitDataView.TestSet, reuseRowObject: true);
            }
        }

        ///<summary>
        ///Use a training data set to produce a Model by featurising the text in the data set
        ///</summary>
        public void Train()
        {
            /*
             * A pipeline is a collection of execution steps used to perform ML training. In this example, there five steps:
             ** Change the name of the Fare column to Label (required for the Algorithm)
             ** Make the VendorID text column numeric
             ** Make the PaymentType text column numeric
             ** Concatenate all numeric columns into a single Features column
             ** Trainer: specifies the ML training alogrithm to use to produce a Model
             * The pipeline is based on 'lazy execution' - these actions are not performed until the Fit method is called
             */

            var pipeline = objContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Fare")
                .Append(objContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIDEncoded", inputColumnName: "VendorID"))
                .Append(objContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(objContext.Transforms.Concatenate("Features", "VendorIDEncoded", "RateCode", "PassengerCount","TripTime","TripDistance","PaymentTypeEncoded"))
                .Append(objContext.Regression.Trainers.FastTree());


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
        ///Produce a set of metrics of a Model by running a set of Test data against it.
        ///</summary>
        public RegressionMetrics Evaluate()
        {
            var predictions = Model.Transform(SplitDataView.TestSet);
            RegressionMetrics metrics = objContext.Regression.Evaluate(predictions, "Label", "Score");
            return metrics;

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
        ///Use a loaded Model to perform a prediction against a new text value.
        ///</summary>
        ///<param name="sample">A pre-populated instance of the RegressionInputSchema containing the values we want to predict against</param>
        public RegressionOutputSchema Predict(RegressionInputSchema sample)
        {

            // Build a prediction Engine by loading the Model
            var predictionEngine = objContext.Model.CreatePredictionEngine<RegressionInputSchema, RegressionOutputSchema>(Model);

            // Generate a prediction against the sampler
            var result = predictionEngine.Predict(sample);

            return result;
        }

    }

    ///<Summary>
    ///The input schema describes the structure of the inbound data set that we want to use for ML
    ///</Summary>
    public class RegressionInputSchema
    {
        [LoadColumn(0), ColumnName("VendorID")]
        public string vendorId { get; set; }

        [LoadColumn(1), ColumnName("RateCode")]
        public float rateCode { get; set; }

        [LoadColumn(2), ColumnName("PassengerCount")]
        public float passengerCount { get; set; }

        [LoadColumn(3), ColumnName("TripTime")]
        public float tripTime { get; set; }

        [LoadColumn(4), ColumnName("TripDistance")]
        public float tripDistance { get; set; }

        [LoadColumn(5), ColumnName("PaymentType")]
        public string paymentType { get; set; }

        [LoadColumn(6), ColumnName("Fare")]
        public float fare { get; set; }

    }

    ///<Summary>
    ///The output schema describes the structure of the Prediction result that will be generated by the Predict method. The structure is dictated by the alogrithm used.
    ///</Summary>
    public class RegressionOutputSchema
    {
        [ColumnName("Score")]
        public float fareAmount;

        //Clean Fare is used for demo purposes, and provides the Fare Amount output in a formatted currency string
        public string cleanFare()
        {
            return $"{fareAmount:$##.##}";
        }

    }
}

