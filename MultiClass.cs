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
    ///Date: 6th June 2019
    ///======================================================================================================================================
    ///MLMultiClassificationModel is an object that can be used to perform Non-Binary Classification Supervised Machine Learning against a Data Set.
    /// Sequence of Events:
    /// 1 - Load the Data
    /// 2 - Train the Model
    /// 3 - Save the Model
    /// 4 - Evaluate the Model (Optional)
    /// 5 - Load the Model (at some point in the future)
    /// 6 - Predict against the Model
    /// This particular implementation uses the SDCA Maximum Entropy Algorithm
    /// 
    /// The Model is designed to perform textual analysis on a DB of support tickets extracted from a project in GitHub.
    /// The tickets are grouped into one of several support Areas. The model attempts to derived patterns in the text to predict which area a future ticket would belong to.    
    ///</summary>      

    public class MLMultiClassificationModel
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
            DataView = objContext.Data.LoadFromTextFile<MultiClassInputSchema>(filePath, hasHeader: false);

            // Split the data into two data sets: Train & Test. Split percentage is based on a parameter provided by the calling program
            SplitDataView = objContext.Data.TrainTestSplit(DataView, testFraction: testPercentage);
        }

        ///<summary>
        ///The Enumerable Data View is a version of the Data Set which can be iterated. Used for demo purposes
        ///<param name="isTrain"/>Determines if the Training Data Set should be used. If False, the Test Set is returned instead</param>
        ///</summary>  
        public IEnumerable<MultiClassInputSchema> enumerableDataView(bool isTrain = true)
        {
            if (isTrain)
            {
                return objContext.Data.CreateEnumerable<MultiClassInputSchema>(SplitDataView.TrainSet, reuseRowObject: true);
            }
            else
            {
                return objContext.Data.CreateEnumerable<MultiClassInputSchema>(SplitDataView.TestSet, reuseRowObject: true);
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
            var pipeline = objContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(objContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(objContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(objContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .Append(objContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(objContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

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
        public MulticlassClassificationMetrics Evaluate()
        {
            var testData = Model.Transform(SplitDataView.TestSet);
            MulticlassClassificationMetrics mcMetrics = objContext.MulticlassClassification.Evaluate(testData);
            return mcMetrics;
        }

        ///<summary>
        ///Use a loaded Model to perform a prediction against a new text value.
        ///</summary>
        ///<param name="sample">An object representing the issue you want to predict the area of</param>
        public MultiClassOutputSchema Predict(MultiClassInputSchema sample)
        {

            // Build a prediction Engine by loading the Model
            var predictionEngine = objContext.Model.CreatePredictionEngine<MultiClassInputSchema, MultiClassOutputSchema>(Model);

            // Generate a prediction against the sampler
            var result = predictionEngine.Predict(sample);

            return result;
        }

    }


    ///<Summary>
    ///The input schema describes the structure of the inbound data set that we want to use for ML
    ///</Summary>
    public class MultiClassInputSchema
    {
        [LoadColumn(0), ColumnName("ID")]
        public string id { get; set; }
        [LoadColumn(1), ColumnName("Area")]
        public string area { get; set; }
        [LoadColumn(2), ColumnName("Title")]
        public string title { get; set; }
        [LoadColumn(3), ColumnName("Description")]
        public string description { get; set; }
    }

    ///<Summary>
    ///The output schema describes the structure of the Prediction result that will be generated by the Predict method. The structure is dictated by the alogrithm used.
    ///</Summary>
    public class MultiClassOutputSchema
    {
        [ColumnName("PredictedLabel")]
        public string area;

        [ColumnName("Score")]
        public Single[] score;

    }
}
