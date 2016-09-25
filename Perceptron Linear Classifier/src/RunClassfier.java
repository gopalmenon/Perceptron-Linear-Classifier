import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class RunClassfier {

	public static final String TRAINING_DATA_FILE = "data/a5a.train";
	public static final String TESTING_DATA_FILE = "data/a5a.test";
	public static final String TRAINING_DATA_FILE_2D = "data/2dTrain.txt";
	public static final String TESTING_DATA_FILE_2D = "data/2dTest.txt";

	public static void main(String[] args) {
		
		try {
			
			LabelsAndFeatures trainingLabelsAndFeatures = DataFileReader.getDataFileContents(TRAINING_DATA_FILE_2D);
			LabelsAndFeatures testingLabelsAndFeatures = DataFileReader.getDataFileContents(TESTING_DATA_FILE_2D);	
			
			PerceptronLinearClassifier classifier = new PerceptronLinearClassifier(true, 1, 1.0, Math.max(getNumberOfFeatures(trainingLabelsAndFeatures.getFeatureVectors()), getNumberOfFeatures(testingLabelsAndFeatures.getFeatureVectors())));
			System.out.println("Number of mistakes made is " + classifier.train(trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors()));
			System.out.println("Weight vector is " + classifier.getWeightVector().toString());
			
			List<Integer> predictions = new ArrayList<Integer>();
			for (String testingDataVector : testingLabelsAndFeatures.getFeatureVectors()) {
				predictions.add(Integer.valueOf(classifier.predict(testingDataVector)));
			}
			
			ClassifierMetrics classifierMetrics = new ClassifierMetrics(testingLabelsAndFeatures.getLabels(), predictions);
			
			System.out.println("Precision: " + classifierMetrics.getPrecision());
			System.out.println("Recall: " + classifierMetrics.getRecall());
			System.out.println("Accuracy: " + classifierMetrics.getAccuracy());
			System.out.println("F1 Score: " + classifierMetrics.getF1Score());
			
		} catch (IOException e) {
			System.err.println("Error while trying to extract labels from test data.");
			e.printStackTrace();
		}

	}
	
	/**
	 * @param featureVectors
	 * @return the number of features in the feature vectors
	 */
	private static int getNumberOfFeatures(List<String> featureVectors) {
		
		int currentLastLabelColumnNumber = 0, numberOfFeatures = 0;

		try {
			for(String featureVector : featureVectors) {
				String[] rawDataColumns = featureVector.split(DataFileReader.WHITESPACE_REGEX);
				String[] lastLabelAndFeature = rawDataColumns[rawDataColumns.length - 1].split(PerceptronLinearClassifier.FEATURE_VALUE_SEPARATOR);
				currentLastLabelColumnNumber = Integer.parseInt(lastLabelAndFeature[0]);
				if (currentLastLabelColumnNumber > numberOfFeatures) {
					numberOfFeatures = currentLastLabelColumnNumber;
				}
			}
		} catch (NumberFormatException e) {
			throw e;
		}
		
		return numberOfFeatures;
	}

}
