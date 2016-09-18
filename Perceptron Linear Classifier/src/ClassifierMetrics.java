import java.util.Iterator;
import java.util.List;

public class ClassifierMetrics {
	
	public static final int POSITIVE_LABEL = 1;

	private double precision;
	private double recall;
	private double accuracy;
	private double f1Score;
	
	public ClassifierMetrics(List<Integer> testingDataLabels, List<Integer> predictions) {
		
		int numberOfTestDataRecords = testingDataLabels.size(), testDataRecordCounter = 0;
		int truePositives = 0, trueNegatives = 0, falsePositives = 0, falseNegatives = 0;
		int testingDataLabel = 0, prediction = 0;
		assert numberOfTestDataRecords == predictions.size() && numberOfTestDataRecords > 0;
	
		Iterator<Integer> testDataIterator = testingDataLabels.iterator();
		while (testDataIterator.hasNext()) {
			
			
			testingDataLabel = testDataIterator.next().intValue();
			prediction = predictions.get(testDataRecordCounter++).intValue();
			
			if (testingDataLabel == POSITIVE_LABEL) {
				if (prediction == testingDataLabel) {
					++truePositives;
				} else {
					++falseNegatives;
				}
			} else {
				if (prediction == testingDataLabel) {
					++trueNegatives;
				} else {
					++falsePositives;
				}
			}
		}
		
		this.precision = (double) truePositives /(truePositives + falsePositives);
		this.recall = (double) truePositives /(truePositives + falseNegatives);
		this.accuracy = (double) (truePositives + trueNegatives) / numberOfTestDataRecords;
		this.f1Score = (double) (2 * truePositives) / (2 * truePositives + falsePositives + falseNegatives);

	}

	public double getPrecision() {
		return precision;
	}

	public double getRecall() {
		return recall;
	}

	public double getAccuracy() {
		return accuracy;
	}

	public double getF1Score() {
		return f1Score;
	}
	
}
