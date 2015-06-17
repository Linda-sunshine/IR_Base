package util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.TreeMap;

public class NumByStringDictionary {
	TreeMap<Integer,String> dictionary;
	int numWords;

	public NumByStringDictionary(String file) throws Exception{
		numWords = 0;
		String line;
		dictionary = new TreeMap<Integer,String>();
		
		BufferedReader fileReader = new BufferedReader(new FileReader(file));
		
	    while((line = fileReader.readLine()) != null){
	    	dictionary.put(numWords++, line.replace("\n", ""));
	    }
	    
	    fileReader.close();
	}
	
	public String getWordString(int word){
		return dictionary.get(word);
	}
}
