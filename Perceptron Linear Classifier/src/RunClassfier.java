import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class RunClassfier {

	public static final String TRAINING_DATA_FILE = "data/a5a.train";
	public static final String TESTING_DATA_FILE = "data/a5a.test";

	public static void main(String[] args) {
		
		try {
			
			LabelsAndFeatures trainingLabelsAndFeatures = DataFileReader.getDataFileContents(TRAINING_DATA_FILE);
			LabelsAndFeatures testingLabelsAndFeatures = DataFileReader.getDataFileContents(TESTING_DATA_FILE);	
			
			PerceptronLinearClassifier classifier = new PerceptronLinearClassifier(true, 1, 1.0, Math.max(trainingLabelsAndFeatures.getFeatureVectors().size(), testingLabelsAndFeatures.getFeatureVectors().size()));
			classifier.train(trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors());
			
			
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

}
