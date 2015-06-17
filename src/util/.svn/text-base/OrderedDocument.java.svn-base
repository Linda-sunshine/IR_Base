package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.List;
import java.util.StringTokenizer;
import java.util.Vector;

public class OrderedDocument extends Document {

	private Vector<Sentence> sentences;
	
	public OrderedDocument() {
		super();
		sentences = new Vector<Sentence>();
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
	
	public static Vector<OrderedDocument> instantiateOrderedDocuments (String path, List<String> authors, List<String> authorList) throws Exception {
		Vector<OrderedDocument> documents = new Vector<OrderedDocument>();
		BufferedReader wordDocFile = new BufferedReader(new FileReader(new File(path)));
		
		int docCount=0;
		String line;
		while(true){
			line = wordDocFile.readLine();
			if(line == null) break;
			StringTokenizer st = new StringTokenizer(line);
			int numSentences = Integer.valueOf(st.nextToken());
			
			OrderedDocument currentDoc = new OrderedDocument();			
			currentDoc.setDocNo(docCount++);
			
			for (int s = 0; s < numSentences; s++) {
				Sentence sentence = new Sentence();
				line = wordDocFile.readLine();
				st = new StringTokenizer(line);
				while(st.hasMoreElements()){
					int wordNo = Integer.valueOf(st.nextToken());
					sentence.addWord(new SentiWord(wordNo));
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
