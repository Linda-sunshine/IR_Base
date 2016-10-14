package Analyzer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;

import opennlp.tools.util.InvalidFormatException;

public class NewMultiThreadedUserAnalyzer extends MultiThreadedUserAnalyzer {

	public NewMultiThreadedUserAnalyzer(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold, int numberOfCores)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores);
		// TODO Auto-generated constructor stub
	}


	// We have different way of loading features.
	@Override
	protected boolean LoadCV(String filename) {
		if (filename==null || filename.isEmpty())
			return false;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			m_Ngram = 1;//default value of Ngram

			while ((line = reader.readLine()) != null) {
				if (line.startsWith("#")){//comments
					if (line.startsWith("#NGram")) {//has to be decoded
						int pos = line.indexOf(':');
						m_Ngram = Integer.valueOf(line.substring(pos+1));
					}						
				} else 
					expandVocabulary(line);
			}
			reader.close();
			System.out.format("Load %d %d-gram features from %s...\n", m_featureNames.size(), m_Ngram, filename);
			m_isCVLoaded = true;
			return true;
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return false;
		}
	}
}
