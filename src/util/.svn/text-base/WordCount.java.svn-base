package util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.text.BreakIterator;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Set;
import java.util.HashMap;
import java.util.Vector;


public class WordCount {
	private static String removePostListFile;// = "RemovePostList.txt";
	private static String removeHostListFile;// = "RemoveHostList.txt";
	private static String makePostListFile;// = "D:/Data/IEEE-Data/Sample-2000-From-Limited-10/SampleList.txt"
	
	private static String stopwordListFile = "stopword.txt";
	private static String dictionaryFile = "Dictionary.txt";
	
	private static String DOC_ADDRESS_IDENTIFIER = "<permalink_>";
	private static String DOC_ADDRESS_END_IDENTIFIER = "</permalink_>";
	private static String CONTENTS_IDENTIFIER = "<content>";
	private static String CONTENTS_END_IDENTIFIER = "</content>";
	private static String DOC_IDENTIFIER = "</doc>";
	
	private static int minimumDocumentLength = 1;
	private static int minimumFrequencyOfWord = 0; 
	private static boolean usingLimitedWords = false;
	private static boolean usingStemmer = true;

	// Do not modify below
	private static BufferedReader fileReader;
	private static BufferedWriter writer;
	private static BufferedWriter wordFreqWriter;
	private static BufferedWriter DxAwriter;
	private static BufferedWriter docWriter;
	private static BufferedWriter wordWriter;
	private static BufferedWriter authorWriter;
	private static BufferedWriter statWriter;
	private static PorterStemmer stemmer = new PorterStemmer();;

	private static Vector<String> stopwords;
	private static Vector<String> removeDocumentList;
	private static Vector<String> removeDocumentListByAuthor;
	private static Vector<String> makeDocumentList;
	private static Vector<String> validWordsList;
	
	private static HashMap<String, Integer> wordDic = new HashMap<String, Integer>();
	private static Vector<Integer> DOCxAUTHOR = new Vector<Integer>();
	private static HashMap<String, Integer> AuthorByNumber = new HashMap<String, Integer>();
	private static HashMap<Integer, Integer> wordSum = new HashMap<Integer, Integer>();
	private static Vector<String> validWordList = new Vector<String>();
	private static Vector<HashMap<Integer, Integer>> documents = new Vector<HashMap<Integer, Integer>>();
	private static HashMap<String, Integer> LimitedBlog = new HashMap<String, Integer>();
	
	private static int doc;
	private static int aut;
	private static int seq;
	private static int remove;
	private static int totalWords;
	private static int word;
	
    private static long start = System.currentTimeMillis();
    
	public static void main(String[] args) throws Exception{
	
		if(args.length != 2){
			System.out.println("Useage java -jar LDA-WordCount.jar inputFile outputDirectory");
			return;
		}
		
		generateBagOfWords(args[0], args[1]);

	}

	private static void generateBagOfWords(String inputFile, String outputDirectory) throws Exception {
		// TODO Auto-generated method stub
		HashMap<Integer, Integer> matchWordDic = new HashMap<Integer, Integer>();

		fileReader = new BufferedReader(new InputStreamReader(new FileInputStream(inputFile)));

		makeOutputFile(outputDirectory);
		initializeList();

		String document= getNextDocument(fileReader);
	    while(document != null){
	    	HashMap<Integer, Integer> documentWordHashTable = new HashMap<Integer, Integer>();
	    	String documentAddress = new String(document.split(DOC_ADDRESS_IDENTIFIER)[1].split(DOC_ADDRESS_END_IDENTIFIER)[0]);
    		String author = documentAddress;//permalink.getHost();
    		String content = document.split(CONTENTS_IDENTIFIER)[1].split(CONTENTS_END_IDENTIFIER)[0].trim().toLowerCase();
    		document = "";
    		
    		//order of code is important!!
    		Vector<String> token = refineDocuments(content);
			
			// remove post when its length is less than minimumDocumentLength
			if(token.size() < minimumDocumentLength || checkValidPost(content, documentAddress, author) == false){
				document = getNextDocument(fileReader);
				continue;
			}
			
			addAuthorInAuthorList(author);
    		
    		Enumeration<String> enumToken = token.elements();
    		while(enumToken.hasMoreElements()){
    			String tmpWord = enumToken.nextElement();
    			
    			addWordToWordDic(tmpWord);
    			addWordToDocument(documentWordHashTable, tmpWord);
    		}
    		
    		if(++doc%1000 == 0){
	    	    float elapsedTimeSec = (System.currentTimeMillis()-start)/1000F;
	    		System.out.println("Parsing Complete - "+doc+" Documents : "+elapsedTimeSec+ " Sec (Elapsed Time)");
    		}
    		
    		documents.add(documentWordHashTable);
    		docWriter.write(documentAddress.toString()+"\n");
    		
    		//get next document
	    	document = getNextDocument(fileReader);
	    }
	    
		docWriter.close();
		authorWriter.close();
		fileReader.close();

		System.out.println("==Parsing Complete==\n");
		
		writeDocumentByAuthor();
		writeWordFrequency(matchWordDic);
		writeMatrixFile(matchWordDic);

		printResult();
		writeResultStat();
	}

	private static void writeMatrixFile(HashMap<Integer, Integer> matchWordDic) throws Exception{
		// TODO Auto-generated method stub
		Enumeration<HashMap<Integer,Integer>> enumDoc = documents.elements();
		
		int cnt = 0;
		
		while(enumDoc.hasMoreElements()){
			int docLength = 0;
			int docWords = 0;
			String temStr = "";
			HashMap<Integer, Integer> tmpDoc = enumDoc.nextElement();
			Enumeration<String> enumValidWord = validWordList.elements();
			
			while(enumValidWord.hasMoreElements()){
				String tmpWord = enumValidWord.nextElement();
				int wordNo = wordDic.get(tmpWord);
				if( tmpDoc.containsKey(wordNo) ){
					int wordCount = tmpDoc.get(wordNo);
					docLength += wordCount;
					docWords++;
					totalWords += wordCount;
					temStr += String.valueOf(matchWordDic.get(wordNo)) + " " + String.valueOf(wordCount) + " " ;
				}
			}
			
			writer.write(docWords + " " +docLength + "\n");
			writer.write(temStr.trim() + "\n");
			if( ++cnt % 1000 == 0){
	    	    // Get elapsed time in milliseconds
	    	    float elapsedTimeSec = (System.currentTimeMillis()-start)/1000F;
				System.out.println("Document Writing -- "+cnt+" Documents Completed : "+elapsedTimeSec+" sec(Elapsed Time)");
			}
		}
		wordWriter.close();
		writer.close();
		
	}

	private static void writeWordFrequency(HashMap<Integer, Integer> matchWordDic) throws Exception {
		// TODO Auto-generated method stub
		Set<String> keys;
		Iterator<String> iter;

		keys = wordDic.keySet();
		iter = keys.iterator();

		while(iter.hasNext()){
			String tmpWord = iter.next();
			int wordNo = wordDic.get(tmpWord);
			if(wordSum.get(wordNo) >= minimumFrequencyOfWord){
				validWordList.add(tmpWord);
				matchWordDic.put(wordNo, word++);
				wordWriter.write(tmpWord+"\n");
				wordFreqWriter.write(tmpWord+","+wordSum.get(wordNo)+"\n");
			}
		}
		wordSum.clear();
		wordFreqWriter.close();
		
	}

	private static void writeDocumentByAuthor() throws Exception {
		// TODO Auto-generated method stub
		// writing author file
		Enumeration<Integer> enumAuthor = DOCxAUTHOR.elements();
		while(enumAuthor.hasMoreElements()){
			DxAwriter.write(enumAuthor.nextElement()+"\n");
		}
		DxAwriter.close();
	}

	private static void addWordToDocument(HashMap<Integer, Integer> DOCxWORDTable, String tmpWord) {
		// TODO Auto-generated method stub
		if(DOCxWORDTable.containsKey(wordDic.get(tmpWord))){
			DOCxWORDTable.put(wordDic.get(tmpWord), DOCxWORDTable.get(wordDic.get(tmpWord))+1);
		}else{
			DOCxWORDTable.put(wordDic.get(tmpWord), 1);
		}
	}

	private static void addWordToWordDic(String tmpWord) {
		// TODO Auto-generated method stub
		if(!wordDic.containsKey(tmpWord)){
			// new vocabulary
			wordDic.put(tmpWord, seq++);	wordSum.put(wordDic.get(tmpWord), 1);
		}else{
			wordSum.put(wordDic.get(tmpWord), wordSum.get(wordDic.get(tmpWord))+1);
		}
		
	}

	private static void addAuthorInAuthorList(String author) throws Exception {
		// TODO Auto-generated method stub
		if(!AuthorByNumber.containsKey(author)){
			// new author's coming
			AuthorByNumber.put(author, aut++);
			authorWriter.write(author+"\n");
		}
		
		// get author's information
		DOCxAUTHOR.add(AuthorByNumber.get(author));
	}

	private static Vector<String> refineDocuments(String content) {
		// TODO Auto-generated method stub
		/* refine document */
		BreakIterator boundary = BreakIterator.getWordInstance();
		boundary.setText(content);
		int startBound=boundary.first();
		Vector<String> token = new Vector<String>();

		for (int endBound = boundary.next(); endBound != BreakIterator.DONE; startBound = endBound, endBound = boundary.next()) {
			boolean validWord = true;
			String tmpWord = new String(content.substring(startBound,endBound));
			char[] tempWord = tmpWord.toCharArray();
			
			if(tempWord.length == 1){
				validWord = false;
			}else{
    			for(int j=0; j<tempWord.length; j++){
    				if( !( (tempWord[j] >= 'a' && tempWord[j] <= 'z') || tempWord[j] == '\'' ) ){
    					validWord = false;
    				}
    			}
			}
			
			if(validWord == true){
				//System.out.println(tmpWord + tempWord.length);
				if(tmpWord.contains("'s")){
					tmpWord = tmpWord.replace("'s", "");
				}
				
				String stemmedWord = tmpWord; 

				if(usingStemmer == true){
					stemmedWord = stemmer.stemming(tmpWord);
				}
				
				if(stopwords.contains(stemmedWord)){
    				validWord = false;
    			}
				
				if(validWord == true){
					
					if(usingLimitedWords  == true && validWordsList.contains(stemmedWord)){
						token.add(stemmedWord);
					}else if(usingLimitedWords == false){
						token.add(stemmedWord);
					}
					
    			}
			}

		}
		/* end refine */
		return token;
	}

	private static void initializeList() throws Exception{
		// TODO Auto-generated method stub

		if(stopwordListFile != null ) stopwords = Utility.makeStemmedVectorFromFile(stopwordListFile, usingStemmer);
		if(dictionaryFile != null  && usingLimitedWords  == true) validWordsList = Utility.makeStemmedValidWordFromFile(dictionaryFile, usingStemmer);
		if(removePostListFile != null ) removeDocumentList = Utility.makeVectorFromFile(removePostListFile);
		if(removeHostListFile != null ) removeDocumentListByAuthor = Utility.makeVectorFromFile(removeHostListFile);
		if(makePostListFile != null ) makeDocumentList = Utility.makeVectorFromFile(makePostListFile);
	}
	
	private static void printResult(){
		System.out.println("Author: "+ aut);
		System.out.println("Word  : "+ word);
		System.out.println("Document: "+ doc);
		System.out.println("TotalWords : "+ totalWords);
		System.out.println("AverageWords : "+ String.format("%.2f",(double)totalWords/doc));
		System.out.println("Removed Posts: "+remove);
	}
	
	private static void writeResultStat() throws Exception{
		statWriter.write("Author = "+ aut +"\n");
		statWriter.write("Word  = "+ word +"\n");
		statWriter.write("Document = "+ doc +"\n");
		statWriter.write("TotalWords = "+ totalWords +"\n");
		statWriter.write("AverageWords = "+ String.format("%.2f",(double)totalWords/doc) +"\n");
		statWriter.close();
	}
	
	public static String getNextDocument(BufferedReader fileReader) throws Exception{
		String line = fileReader.readLine();
		String document = line;
		
		if(line == null){
			fileReader.close();
			return null;
		}
		
		while(!line.contains(DOC_IDENTIFIER)){
			document += line;
			
			if((line = fileReader.readLine()) == null){
				fileReader.close();
				return null;
			}
		}
		
		return document;
	}
	
	private static boolean checkValidPost(String content, String documentAddr, String author) {
		// TODO Auto-generated method stub
		/* make a bag of word only permalink which exist in make list */ 
//		if( makeDocumentList != null && !makeDocumentList.contains(author.toLowerCase())){
//			return false;
//		}
//		
//		/* remove posts in the remove post list */
//
//		if( removeDocumentList != null && removeDocumentList.contains(author.toLowerCase())){
//			removeDocumentList.remove(author.toLowerCase());
//			remove++;
//			return false;
//		}
//		
//		if( removeDocumentListByAuthor != null && removeDocumentListByAuthor.contains(author.toLowerCase())){
//			remove++;
//			return false;
//		}

		/* Each author(host)'s posts are limited by 10*/

//		String host = permalink.getHost();
//		
//		if(LimitedBlog.containsKey(host)){
//			if(LimitedBlog.get(host) > 10){
//				document = getNextDocument(fileReader);
//				continue;
//			}else{
//				LimitedBlog.put(host, LimitedBlog.get(host)+1);
//			}
//			
//		}else{
//			LimitedBlog.put(host, 1);
//		}
		
		return true;
	}
	
	private static void makeOutputFile(String outputDirectory) throws Exception{
		docWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDirectory+"/DocumentList.txt")));
		statWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDirectory+"/Statistic.txt")));
		authorWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDirectory+"/AuthorList.txt")));
		wordWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDirectory+"/WordList.txt")));
		DxAwriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDirectory+"/DOCxAUTHOR.txt")));
		wordFreqWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDirectory+"/WORDxFREQ.csv")));
		writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputDirectory+"/WORDxDOC.txt")));
	}
	
}



