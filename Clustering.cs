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
    ///MLClusteringModel is an object that can be used to perform Clustered Unsupervised Machine Learning against a Data Set.
    /// Sequence of Events:
    /// 1 - Load the Data
    /// 2 - Train the Model
    /// 3 - Save the Model
    /// 4 - Load the Model (at some point in the future)
    /// 5 - Predict against the Model
    /// This particular implementation uses the K-Means Clustering Algorithm
    ///     
    /// The Model is designed to perform clustering of the famous Fisher's Iris Data Set, which describe numeric values of a collection of Iris flowers.
    /// The data set already groups the flowers into one of three species of Iris.
    /// The objective is to use Clustered unsupervised learning to automatically determine the family (Cluster) of each flower, and then use this model
    /// to be able to predict the cluster of a new value. The label is ignored, although it can be used retrospectively to prove the model.
    ///</summary>   
    public class MLClusteringModel
    {
        private readonly MLContext objContext = new MLContext();

        private IDataView dataView { get; set; }
        private ITransformer Model { get; set; }

        ///<summary>
        ///Create a data view of a data file. The data must be comma-delimited and must be in the format as laid out in the ClusteringInputSchema class.
        ///</summary>
        ///<param name="filePath">The local file-system path of the data set to import</param>
        public void LoadData(string filePath)
        {
            // Build a data view with the contents of the file - the file schema must match the schema laid out in the InputSchema
            dataView = objContext.Data.LoadFromTextFile<ClusteringInputSchema>(filePath, separatorChar: ',', hasHeader: false);

        }

        ///<summary>
        ///The Enumerable Data View is a version of the Data Set which can be iterated. Used for demo purposes
        ///</summary>        
        public IEnumerable<ClusteringInputSchema> enumerableDataView()
        {
            return objContext.Data.CreateEnumerable<ClusteringInputSchema>(dataView, reuseRowObject: true);

        }

        ///<summary>
        ///Generate a Machine Learning Model by concatenating the input data (minus the label) into a Features column and passing it through a K-Means algorithm
        ///</summary>   
        public void Train()
        {
            string featuresColumnName = "Features";
            var pipeline = objContext.Transforms
                .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(objContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

            Model = pipeline.Fit(dataView);
        }

        ///<summary>
        ///Output the completed model to the file system.
        ///</summary>
        ///<param name="filepath">The local file-system path to store the Model. Should include a .zip file extension</param>
        public void Save(string filepath)
        {
            objContext.Model.Save(Model, dataView.Schema, filepath);
        }

        ///<summary>
        ///Use a loaded Model to predict which Cluster a new instance of the input values belong to.
        ///</summary>
        ///<param name="sample">A pre-populated instance of the ClusteringInputSchema containing the values we want to predict against</param>
        public ClusteringOutputSchema Predict(ClusteringInputSchema sample)
        {
            var predictor = objContext.Model.CreatePredictionEngine<ClusteringInputSchema, ClusteringOutputSchema>(Model);
            return predictor.Predict(sample);
        }

        ///<summary>
        ///Load a previously generated Model from the file system.
        ///</summary>
        ///<param name="modelfilepath">The local file-system path of the Model to load</param>
        public void Load(string modelfilepath)
        {
            DataViewSchema modelSchema;
            Model = objContext.Model.Load(modelfilepath, out modelSchema);

        }


    }

    ///<Summary>
    ///The input schema describes the structure of the inbound data set that we want to use for ML
    ///</Summary>
    public class ClusteringInputSchema
    {
        [LoadColumn(0), ColumnName("PetalLength")]
        public Single petalLength { get; set; }

        [LoadColumn(1), ColumnName("PetalWidth")]
        public Single petalWidth { get; set; }

        [LoadColumn(2), ColumnName("SepalLength")]
        public Single sepalLength { get; set; }

        [LoadColumn(3), ColumnName("SepalWidth")]
        public Single sepalWidth { get; set; }

        //[LoadColumn(4), ColumnName("Type")]
        //public string type { get; set; }

        ///<Summary>
        ///FlattenedText is for demo purposes, and provides a delimited string containing all values of the Class
        ///</Summary>
        public string flattenedText()
        {
            return petalLength.ToString() + "," + petalWidth.ToString() + "," + sepalLength.ToString() + "," + sepalWidth.ToString();
        }
    }

    ///<Summary>
    ///The output schema describes the structure of the Prediction result that will be generated by the Predict method. The structure is dictated by the alogrithm used.
    ///</Summary>
    public class ClusteringOutputSchema
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedCluster;

        [ColumnName("Score")]
        public float[] Distances;

        ///<Summary>
        ///FlattenedDistances is for demo purposes, and provides a formatted String containing the distances the predicted value is from the centre of each Cluster
        ///</Summary>
        public string flattenedDistances()
        {
            Int16 i = 1;
            StringBuilder output = new StringBuilder();            
            foreach(float f in Distances)
            {
                output.Append(" (Cluster: ");
                output.Append(i.ToString());
                output.Append(" Distance: ");
                output.Append(f.ToString());
                output.Append(") ");
                i++;
            }

            return output.ToString();
        }

    }
}
