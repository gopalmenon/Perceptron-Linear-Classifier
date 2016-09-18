
public class RunClassfier {

	public static final String TRAINING_DATA_FILE = "data/a5a.train";
	public static final String TESTING_DATA_FILE = "data/a5a.test";

	public static void main(String[] args) {
		
		PerceptronLinearClassifier classifier = new PerceptronLinearClassifier(true, 1, TRAINING_DATA_FILE, TESTING_DATA_FILE);
		classifier.train();
	}

}
