package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.TreeSet;

public class Stemming {

	public static void main(String [] args) throws Exception {
		String inputFileName = args[0];
		String outputFileName = args[1];
		
		PorterStemmer stemmer = new PorterStemmer();
		
		BufferedReader inputFile = new BufferedReader(new FileReader(new File(inputFileName)));
		
		TreeSet<String> wordSet = new TreeSet<String>();
		String line;
		while ((line = inputFile.readLine()) != null) {
			if (line.trim() == "") continue;
			String word = line.trim().replaceFirst("[\\W].*", "").toLowerCase();
			String stemmedWord = stemmer.stemming(word);
			wordSet.add(stemmedWord);
		}
		
		inputFile.close();
		
		PrintWriter outputFile = new PrintWriter(new FileWriter(new File(outputFileName)));
		for (String word : wordSet) {
			outputFile.println(word);
		}
		outputFile.close();
	}
}
