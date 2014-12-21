package Classifier;

import java.util.ArrayList;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.Normalizer;
import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;

public class test {
	
	private Tokenizer m_tokenizer;
	private SnowballStemmer m_stemmer;
	
	public test() throws InvalidFormatException, FileNotFoundException, IOException{
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream("./data/Model/en-token.bin")));
		m_stemmer = new englishStemmer();
	}
	
	public String[] Tokenizer(String source){
		String[] tokens = m_tokenizer.tokenize(source);
		return tokens;
	}
	
	//Normalize.
	public String Normalize(String token){
		token = Normalizer.normalize(token, Normalizer.Form.NFKC);
		token = token.replaceAll("\\W+", "");
		token = token.toLowerCase();
		return token;
	}
	
	//Snowball Stemmer.
	public String SnowballStemming(String token){
		m_stemmer.setCurrent(token);
		if(m_stemmer.stem())
			return m_stemmer.getCurrent();
		else
			return m_stemmer.getCurrent();
	}
	
	public String[] TokenizerNormalizeStemmer(String source){
		String[] tokens = Tokenizer(source); //Original tokens.
		//Normalize them and stem them.		
		for(int i = 0; i < tokens.length; i++)
			tokens[i] = SnowballStemming(Normalize(tokens[i]));
		
		int tokenLength = tokens.length, N = 3, NgramNo = 0;
		ArrayList<String> Ngrams = new ArrayList<String>();
		
		//Collect all the grams, Ngrams, N-1grams...
		while (N > 0) {
			NgramNo = tokenLength - N + 1;
			StringBuffer Ngram = new StringBuffer(128);
			for (int i = 0; i < NgramNo; i++) {
				Ngram.setLength(0); 
				if (!tokens[i].equals("")) {
					for (int j = 0; j < N; j++) {
						if (j == 0)
							Ngram.append(tokens[i + j]);
						else {
							if (!tokens[i + j].equals("")) {
								Ngram.append("-" + tokens[i + j]);
							} else {
								Ngram.setLength(0);
								break;
							}
						}
					}
				}
				if (!(Ngram.length() == 0)) {Ngrams.add(Ngram.toString());}
			} N--;
		}
		return Ngrams.toArray(new String[Ngrams.size()]);
	}
	
	public static void main(String [] args) throws InvalidFormatException, FileNotFoundException, IOException{
		System.out.println("testing the git config ...");
		test mytest = new test();
		String test = "I am Lin,I like sunshine, sea, sky.";
		String[] res = mytest.TokenizerNormalizeStemmer(test);
		for(String r: res){
			System.out.println(r);
		}
	}
}
