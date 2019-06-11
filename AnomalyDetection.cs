using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Data;
using System.Collections.Generic;

namespace MLDemo
{
    ///<summary> 
    ///Author: John Ferguson
    ///Date: 5th June 2019
    ///Date: 5th June 2019
    ///======================================================================================================================================
    ///MLAnomalyDetectionModel is an object that can be used to perform Anomaly Detection against a Time-Series Data Set.
    /// Sequence of Events:
    /// 1 - Load the Data
    /// 2 - Train the Model
    /// 3 - Analyse the Results
    /// 
    /// The Model is designed to detect spikes in a series of data
    ///</summary> 

    public class MLAnomalyDetectionModel
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
        public void LoadData(string filePath)
        {
            // Build a data view with the contents of the file - the file schema must match the schema laid out in ClassificationInputSchema
            DataView = objContext.Data.LoadFromTextFile<AnomalyDetectionInputSchema>(filePath, separatorChar: ',', hasHeader: true);

        }

        ///<summary>
        ///The Enumerable Data View is a version of the Data Set which can be iterated. Used for demo purposes
        ///<param name="isTrain"/>Determines if the Training Data Set should be used. If False, the Test Set is returned instead</param>
        ///</summary>  
        public IEnumerable<AnomalyDetectionInputSchema> enumerableDataView()
        {

            return objContext.Data.CreateEnumerable<AnomalyDetectionInputSchema>(DataView, reuseRowObject: true);
 
        }

        ///<summary>
        ///Use a training data set to produce a Model by featurising the text in the data set       
        ///</summary>
        ///<param name="confidence">The minimum percentage the P-Value must be in order to trigger an alert</param>///
        ///<param name="docSize">The approximate number of records in the data set. Used to determine the precision of the prediction</param>
        ///<param name="sensitivity">The higher the sensitivity, the shallower the difference in items is needed to trigger a spike detection</param>        
        public void Train(int confidence = 95, int sensitivity = 4, int docSize = 40)
        {            
             
            var pipeline = objContext.Transforms.DetectIidSpike(outputColumnName: nameof(AnomalyDetectionOutputSchema.Prediction), inputColumnName: nameof(AnomalyDetectionInputSchema.NumSales), confidence: confidence, pvalueHistoryLength:docSize / sensitivity);
            Model = pipeline.Fit(DataView);
        }


        ///<summary>
        ///Output the completed model to the file system.
        ///</summary>
        ///<param name="filepath">The local file-system path to store the Model. Should include a .zip file extension</param>
        public void Save(string filepath)
        {
            objContext.Model.Save(Model, DataView.Schema, filepath);
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
        public AnomalyDetectionMetrics Evaluate()
        {
            var testData = Model.Transform(DataView);
            AnomalyDetectionMetrics adMetrics = objContext.AnomalyDetection.Evaluate(testData);
            return adMetrics;
        }

        public IEnumerable<AnomalyDetectionOutputSchema> DetectSpike()
        {
            var predData = Model.Transform(DataView);
            var predictions = objContext.Data.CreateEnumerable<AnomalyDetectionOutputSchema>(predData, reuseRowObject: false);

            return predictions;
        }

    }

    ///<Summary>
    ///The input schema describes the structure of the inbound data set that we want to use for ML
    ///</Summary>
    public class AnomalyDetectionInputSchema
    {
        [LoadColumn(0), ColumnName("Day")]
        public string Day;

        [LoadColumn(1),ColumnName("NumSales")]
        public float NumSales;
    }

    ///<Summary>
    ///The output schema describes the structure of the Prediction result that will be generated by the Predict method. The structure is dictated by the alogrithm used.
    ///</Summary>
    public class AnomalyDetectionOutputSchema
    {
        //vector holds three values; alert, score, p-value
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
