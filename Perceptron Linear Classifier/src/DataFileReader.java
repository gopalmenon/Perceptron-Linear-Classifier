import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataFileReader {
	
	public static String WHITESPACE_REGEX = "\\s";


	/**
	 * @param filePath
	 * @return labels and features 
	 * @throws IOException
	 */
	public static LabelsAndFeatures getDataFileContents(String filePath) throws IOException {
		
		List<Integer> labels = new ArrayList<Integer>();
		List<String> featureVectors = new ArrayList<String>();
		BufferedReader bufferedReader = null;
		
		try {
			bufferedReader = new BufferedReader(new FileReader(filePath));
			String fileLine = bufferedReader.readLine();
			
			while (fileLine != null) {
				
				if (fileLine.trim().length() > 0) {
					
					labels.add(Integer.parseInt(fileLine.trim().split(WHITESPACE_REGEX)[0]));
					featureVectors.add(fileLine.trim().substring(fileLine.trim().indexOf(' ') + 1));
				}
				fileLine = bufferedReader.readLine();
			}
			bufferedReader.close();
		} catch (IOException e) {
			throw e;
		} catch (NumberFormatException e) {
			throw e;
		}
		
		return new LabelsAndFeatures(labels, featureVectors);
		
	}

}
