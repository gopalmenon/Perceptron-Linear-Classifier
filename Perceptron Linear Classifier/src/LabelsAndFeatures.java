import java.util.List;


/**
 *This class will store labels and features as separate lists
 *
 */
public class LabelsAndFeatures {

	private List<Integer> labels;
	private List<String> featureVectors;
	
	//Constructor
	public LabelsAndFeatures(List<Integer> labels, List<String> featureVectors) {
		this.labels = labels;
		this.featureVectors = featureVectors;
	}
	
	//Getters
	public List<Integer> getLabels() {
		return labels;
	}

	public List<String> getFeatureVectors() {
		return featureVectors;
	}

}
