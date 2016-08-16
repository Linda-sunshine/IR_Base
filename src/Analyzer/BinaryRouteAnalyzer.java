package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures.TokenizeResult;
import structures._Doc;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

public class BinaryRouteAnalyzer extends UserAnalyzer {

	public BinaryRouteAnalyzer(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
	}

	@Override
	// Load one file as a user here. 
	public void loadOneUser(String filename){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;			
			String userID = extractUserID(file.getName()); //UserId is contained in the filename.
			// Skip the first line since it is not instances.
			reader.readLine(); 
			
			int ylabel;	
			String[] strs;
			_Review review;
			ArrayList<_Review> reviews = new ArrayList<_Review>();
			while((line = reader.readLine()) != null){
				strs = line.split("\\s+");
				if(strs.length == 15){
					// Construct the new review.
					ylabel = Integer.valueOf(strs[14]);
					review = new _Review(m_corpus.getCollection().size(), line, ylabel);
					AnalyzeDoc(review);
					reviews.add(review);
					m_corpus.addDoc(review);
					m_classMemberNo[ylabel]++;
				}
			}
			allocateReviews(reviews);			
			m_users.add(new _User(userID, m_classNo, reviews)); //create new user from the file.
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	/*Analyze a document and add the analyzed document back to corpus.*/
	@Override
	protected boolean AnalyzeDoc(_Doc review) {
		String[] strs = review.getSource().split("\\s+");
		_SparseFeature[] fvs = new _SparseFeature[strs.length-1];

		for(int i=0; i<strs.length-1; i++)
			fvs[i] = new _SparseFeature(i, Double.valueOf(strs[i]));
		review.setSpVct(fvs);
		return true;
	}
	
	// Normalize
	public void Normalize(int norm){
		if (norm == 1){
			for(_Doc d: m_corpus.getCollection())			
				Utils.L1Normalization(d.getSparse());
		} else if(norm == 2){
			for(_Doc d: m_corpus.getCollection())			
				Utils.L2Normalization(d.getSparse());
		}
	}
}
