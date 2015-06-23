package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.List;
import java.util.StringTokenizer;
import java.util.Vector;

public class OrderedDocument extends Document {
	
	public int label;
	private Vector<Sentence> sentences;
//	private static HashMap<Integer, String> indexes;
	
	public void setLabel(int l){
		label = l;
	}
	
	public OrderedDocument() {
		super();
		sentences = new Vector<Sentence>();
//		indexes = new HashMap<Integer, String>();
	}
	
	public void addWord(Word word) {
		super.addWord(word);
		sentences.lastElement().addWord(word);
	}
	
	public void addWord(int wordNo) {
		addWord(new Word(wordNo));
	}
	
	public void addSentence(Sentence sentence) {
		sentences.add(sentence);
		for (Word word : sentence.getWords())
			words.add(word);
	}
	
	public Vector<Sentence> getSentences() {
		return sentences;
	}
	
//	public HashMap<Integer, String> returnIndexes(){
//		return indexes;
//	}
	public static Vector<OrderedDocument> instantiateOrderedDocuments (String path, List<String> authors, List<String> authorList) throws Exception {
		Vector<OrderedDocument> documents = new Vector<OrderedDocument>();
		BufferedReader wordDocFile = new BufferedReader(new FileReader(new File(path)));
		
		int docCount=0;
		String line;
		while(true){
			line = wordDocFile.readLine();
			if(line == null) break;
			if(line.startsWith("#"))
				line = line.substring(1);
			StringTokenizer st = new StringTokenizer(line);
			String reviewID = st.nextToken();
			int numSentences = Integer.valueOf(st.nextToken());
			int label = Integer.valueOf(st.nextToken());
		
			OrderedDocument currentDoc = new OrderedDocument();		
			currentDoc.setReviewID(reviewID);
			currentDoc.setDocNo(docCount++);
			currentDoc.setLabel(label);
//			indexes.put(currentDoc.getDocNo(), reviewID);
			
			for (int s = 0; s < numSentences; s++) {
				Sentence sentence = new Sentence();
				line = wordDocFile.readLine();
//				//System.out.println(line);
				st = new StringTokenizer(line);
//				sentence.label = Integer.valueOf(st.nextToken());
//				if(sentence.label==-1)
//				{
//					sentence.label = 0;
//					//System.out.println("pros");
//				}
//				else if(sentence.label==-2)
//				{
//				 sentence.label = 1;
//				 //System.out.println("cons");
//				} else{
//					sentence.addWord(new SentiWord(sentence.label));
//				}
				//&& !st.nextToken().startsWith("#")
				while(st.hasMoreElements()){
					String tmp = st.nextToken();
					if(!tmp.startsWith("#")){
						int wordNo = Integer.valueOf(tmp);
						//System.out.println(wordNo);
						sentence.addWord(new SentiWord(wordNo));
					} else break;
				}
				currentDoc.addSentence(sentence);
			}
			
			if (authors != null) {
				int author = Integer.valueOf(authorList.indexOf(authors.get(currentDoc.getDocNo())) );
				currentDoc.setAuthor(author);
			}
			
			documents.add(currentDoc);
		}
		wordDocFile.close();
		
		return documents;
	}
	
}
