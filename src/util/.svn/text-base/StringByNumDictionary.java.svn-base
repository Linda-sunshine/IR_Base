package util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.TreeMap;

public class StringByNumDictionary {
	
	TreeMap<String, Integer> dictionary;
	int numWords;

	public StringByNumDictionary(String file) throws Exception{
		numWords = 0;
		String line;
		dictionary = new TreeMap<String, Integer>();
		
		BufferedReader fileReader = new BufferedReader(new FileReader(file));
	    while((line = fileReader.readLine()) != null){
	    	dictionary.put(line.replace("\n", ""), numWords++);
	    }
	    
	    fileReader.close();
	    
	}
	
	public int getWordNum(String word){
		return dictionary.get(word);
	}
	
	public boolean containsWord(String word){
		return dictionary.containsKey(word);
	}
}
