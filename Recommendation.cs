using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Data;
using System.Linq;

namespace MLDemo
{
    ///<summary> 
    ///Author: John Ferguson
    ///Date: 6th June 2019
    ///======================================================================================================================================
    ///RecommendationModel is an object that can be used to perform Recommendatiomn-based Supervised Machine Learning against a Data Set.
    /// Sequence of Events:
    /// 1 - Load the Data
    /// 2 - Train the Model
    /// 3 - Save the Model
    /// 4 - Evaluate the Model (Optional)
    /// 5 - Load the Model (at some point in the future)
    /// 6 - Generate Recommendations (incorporating an iteration of Prediction)
    /// This particular implementation uses the Matrix Factorisation Algorithm, which is an example of Regression
    /// 
    /// The Model is designed to generate a set of recommendations based on how the user, and other users have rated similar groups of things in the past.
    /// This specific example uses a list of movies which have been rated by users. Each user gives a movie they have seen a rating out of 5.
    /// The model looks for patterns in the user's ratings, incorporating other ratings from other users within similar 'rating clusters'.
    /// The model can be then used to predict what a user would rate a movie they haven't seen, and in turn, to generate a list of recommended movies.
    ///</summary>    
    public class MLRecommendationModel
    {
        private readonly MLContext objContext = new MLContext();

        private IDataView DataView { get; set; }
        private ITransformer Model { get; set; }
        private TrainTestData SplitDataView { get; set; }

        // The reference list of movie ID and titles - not used in Machine Learning but required for recommendation lists
        private IDictionary<int, string> MovieList { get; set; }

        // The list of movies the user has already seen; we don't want to inlcude these in the recommendations
        private IDictionary<string, bool> HasSeenList { get; set; }


        ///<summary>
        ///Create a training & test data set given a valid path of a data file. The data is expected to be comma-delimited and must be in the format as laid out in the RecommendationInputSchema class.
        ///</summary>
        ///<param name="filePath">The local file-system path of the data set to import</param>
        ///<param name="testPercentage">The proportion of the dataset to be used for Testing (as opposed to Training)</param>     
        public void LoadData(string filePath, string referencePath, double testPercentage = 0.2)
        {
            // Build a data view with the contents of the file - the file schema must match the schema laid out in RecommendationInputSchema
            DataView = objContext.Data.LoadFromTextFile<RecommendationInputSchema>(filePath, separatorChar: ',', hasHeader: true);

            // Build a list of movies each user has seen
            GenerateHasSeenList();

            SplitDataView = objContext.Data.TrainTestSplit(DataView, testFraction: testPercentage);

            // Create a reference list of movie IDs and Names
            IDataView referenceData = objContext.Data.LoadFromTextFile<Movie>(referencePath, separatorChar: ',', hasHeader: true);
            BuildReferenceDictionary(referenceData);

        }

        ///<summary>
        ///The Enumerable Data View is a version of the Data Set which can be iterated. Used for demo purposes
        ///<param name="isTrain"/>Determines if the Training Data Set should be used. If False, the Test Set is returned instead</param>
        ///</summary>  
        public IEnumerable<RecommendationInputSchema> enumerableDataView(bool isTrain = true)
        {
            if (isTrain)
            {
                return objContext.Data.CreateEnumerable<RecommendationInputSchema>(SplitDataView.TrainSet, reuseRowObject: true);
            }
            else
            {
                return objContext.Data.CreateEnumerable<RecommendationInputSchema>(SplitDataView.TestSet, reuseRowObject: true);
            }
        }

        ///<summary>
        ///The HasSeenList is a dictionary which is used to hold a list of each User / Movie pair. This is later used to remove movies the 
        ///user has already seen from their list of recommendations
        ///</summary>  
        private void GenerateHasSeenList()
        {
            HasSeenList = new Dictionary<string, bool>();

            IEnumerable<RecommendationInputSchema> seenList = objContext.Data.CreateEnumerable<RecommendationInputSchema>(DataView, reuseRowObject: true);
            foreach (RecommendationInputSchema i in seenList)
            {
                // The Key for the HasSeenList is a concatenation of the User ID & Movie ID
                HasSeenList.Add(i.userId.ToString() + ":" + i.movieId.ToString(), true);
            }

        }

        ///<summary>
        ///The Reference Dictionary is a list of Movie IDs and Titles. We don't need the titles for Machine Learning, but we do need them to display
        ///the recommendations to the user at the end.
        ///</summary> 
        private void BuildReferenceDictionary(IDataView referenceData)
        {
            MovieList = new Dictionary<int, string>();

            IEnumerable<Movie> refList = objContext.Data.CreateEnumerable<Movie>(referenceData, reuseRowObject: true);
            foreach (Movie m in refList)
            {
                MovieList.Add(m.movieId, m.title);
            }

        }

        ///<summary>
        ///Use a training data set to produce a Model
        ///<param name="iterations">The number of times to loop through the alogorithm. Higher values will produce a more accurate model, but will take longer</param>
        ///</summary>
        public void Train(int iterations = 10)
        {

            var recommenderOptions = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Rating",
                NumberOfIterations = iterations,
                ApproximationRank = 500,
                LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossRegression
            };

            /*
             * A pipeline is a collection of execution steps used to perform ML training. In this example, there are three steps:
             ** Convert the userID & MovieID values into Matrix keys
             ** Trainer: specifies the ML training alogrithm to use to produce a Model
             * The pipeline is based on 'lazy execution' - these actions are not performed until the Fit method is called
             */
            var pipeline = objContext.Transforms.Conversion.MapValueToKey(inputColumnName: "UserID", outputColumnName: "userIdEncoded")
                .Append(objContext.Transforms.Conversion.MapValueToKey(inputColumnName: "MovieID", outputColumnName: "movieIdEncoded")
                .Append(objContext.Recommendation().Trainers.MatrixFactorization(recommenderOptions)));

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
        public void Load(string modelfilepath, string refpath, string datafilepath = null)
        {
            DataViewSchema modelSchema;
            Model = objContext.Model.Load(modelfilepath, out modelSchema);
            if (datafilepath != null)
            {
                LoadData(datafilepath, refpath);
            }
        }

        ///<summary>
        ///Produce a set of metrics of a Model by running a set of Test data against it.
        ///</summary>
        public RegressionMetrics Evaluate()
        {
            var testData = Model.Transform(SplitDataView.TestSet);            
            RegressionMetrics regMetrics = objContext.Recommendation().Evaluate(data: testData, labelColumnName: "Rating",scoreColumnName: "Score");
            return regMetrics;
        }


        ///<summary>
        ///Use a loaded Model to perform a prediction against a new text value; in this case - predicting the rating a particular user would give a movie
        ///</summary>
        ///<param name="sample">A user ID and movie ID combination.</param>
        public RecommendationOutputSchema Predict(RecommendationInputSchema sample)
        {

            // Build a prediction Engine by loading the Model
            var predictionEngine = objContext.Model.CreatePredictionEngine<RecommendationInputSchema, RecommendationOutputSchema>(Model);

            // Generate a prediction against the sampler
            var result = predictionEngine.Predict(sample);

            return result;
        }

        ///<summary>
        ///Generate an ordered list of movies to recommend to a user, based on the predicted rating that the user would give the movie
        ///</summary>
        ///<param name="userId">The user to perform the recommendations for.</param>
        ///<param name="count">The number of recommendations we want to appear in the list</param>
        ///
        public List<MovieRecommendation> GetRecommendations(int userId, int count)
        {
            IDictionary<int, float> ratings = new Dictionary<int, float>();
            RecommendationOutputSchema pred;

            // Iterate through the full list of movies
            foreach (KeyValuePair<int, string> entry in MovieList)
            {
                //ignore movies the user has already seen
                if (!HasSeenList.TryGetValue(userId.ToString() + ":" + entry.Key.ToString(), out bool isSeen))
                { 
                    //predict the rating the user would give the movie
                    pred = Predict(new RecommendationInputSchema { userId = userId, movieId = entry.Key });

                    //Add it to the list
                    ratings.Add(entry.Key, pred.Score);
                }
            }

           //Sort the list by predicted rating, in descending order
            List<KeyValuePair<int, float>> ratingList = ratings.ToList();
            ratingList.Sort(new Comparison<KeyValuePair<int, float>>(CompareKvp));
            
            var Predictions = new List<MovieRecommendation>();
            Int16 iter = 0;
         
            //Loop through the ordered list
            foreach (KeyValuePair<int, float> predictedRating in ratingList)
            {
                //Create a new recommendation record
                var recommendation = new MovieRecommendation();
                recommendation.movieId = predictedRating.Key;
                recommendation.score = predictedRating.Value;

                //Get the title of the movie from the reference list
                MovieList.TryGetValue(recommendation.movieId, out recommendation.title);
                Predictions.Add(recommendation);
                iter++;

                //When we hit the recommendation count, stop
                if (iter == count)
                {
                    break;
                }
            }

            return Predictions;


        }

        ///<summary>
        ///A utility method to allow us to order the Predicted Ratings list in descending order
        ///</summary>
        private static Int32 CompareKvp(KeyValuePair<int, float> kvp1, KeyValuePair<int, float> kvp2)
        {
            int ret = -kvp1.Value.CompareTo(kvp2.Value); // negate for descending
            if (ret == 0)
            {
                ret = kvp1.Key.CompareTo(kvp2.Key); // Ascend comp for Key
            }
            return ret;
        }


    }

    ///<Summary>
    ///The input schema describes the structure of the inbound data set that we want to use for ML
    ///</Summary>
    public class RecommendationInputSchema
    {
        [LoadColumn(0), ColumnName("UserID")]
        public float userId;

        [LoadColumn(1), ColumnName("MovieID")]
        public float movieId;

        [LoadColumn(2), ColumnName("Rating")]
        public float rating;

    }

    ///<Summary>
    ///The output schema describes the structure of the Prediction result that will be generated by the Predict method. The structure is dictated by the alogrithm used.
    ///</Summary>
    public class RecommendationOutputSchema
    {
        public float Label;
        public float Score;
    }


    ///<Summary>
    ///The input schema of the Movie reference list
    ///</Summary>
    public class Movie
    {
        [LoadColumn(0), ColumnName("MovieID")]
        public Int32 movieId;

        [LoadColumn(1), ColumnName("Title")]
        public string title;

    }

    ///<Summary>
    ///The schema of a Movie Recommendation which will be presented back to the user
    ///</Summary>
    public class MovieRecommendation
    {
        public Int32 movieId;
        public float score;
        public string title;

    }



}
