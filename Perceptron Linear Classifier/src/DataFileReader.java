import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataFileReader {
	
	/**
	 * @param filePath
	 * @return a list of strings having file contents
	 * @throws IOException
	 */
	public static List<String> getDataFileContents(String filePath) throws IOException {
		
		List<String> dataFileContents = new ArrayList<String>();
		BufferedReader bufferedReader = null;
		
		try {
			bufferedReader = new BufferedReader(new FileReader(filePath));
			String fileLine = bufferedReader.readLine();
			
			while (fileLine != null) {
				
				if (fileLine.trim().length() > 0) {
					dataFileContents.add(fileLine.trim());
				}
				fileLine = bufferedReader.readLine();
			}
			bufferedReader.close();
		} catch (IOException e) {
			throw e;
		}
		
		return dataFileContents;
		
	}
	
	/**
	 * @param filePath
	 * @return a list of label values which will be the first column in the data
	 * @throws IOException
	 */
	public static List<Integer> getLabelValues(String filePath) throws IOException {
		
		List<Integer> labelValues = new ArrayList<Integer>();
		BufferedReader bufferedReader = null;
		
		try {
			
			bufferedReader = new BufferedReader(new FileReader(filePath));
			String fileLine = bufferedReader.readLine();
			
			while (fileLine != null) {
				
				if (fileLine.trim().length() > 0) {
					labelValues.add(Integer.parseInt(fileLine.trim().split("\\s")[0]));
				}
				fileLine = bufferedReader.readLine();
			}
			
			bufferedReader.close();
			
		} catch (IOException e) {
			throw e;
		} catch (NumberFormatException e) {
			throw e;
		}
		
		return labelValues;
		
	}

}
