import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class RunClassfier {

	public static final String TRAINING_DATA_FILE = "data/a5a.train";
	public static final String TESTING_DATA_FILE = "data/a5a.test";

	public static void main(String[] args) {
		
		PerceptronLinearClassifier classifier = new PerceptronLinearClassifier(true, 1, 1.0, TRAINING_DATA_FILE, TESTING_DATA_FILE);
		classifier.train();
		
		try {
			List<Integer> testingDataLabels = DataFileReader.getLabelValues(TESTING_DATA_FILE);
			List<String> testingData = DataFileReader.getDataFileContents(TESTING_DATA_FILE);
			
			List<Integer> predictions = new ArrayList<Integer>();
			for (String testingDataRow : testingData) {
				predictions.add(Integer.valueOf(classifier.predict(testingDataRow)));
			}
			
			ClassifierMetrics classifierMetrics = new ClassifierMetrics(testingDataLabels, predictions);
			
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
