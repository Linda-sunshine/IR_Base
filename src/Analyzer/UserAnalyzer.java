package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import opennlp.tools.util.InvalidFormatException;
import structures._Review;
import structures._User;

public class UserAnalyzer extends DocAnalyzer {
	
	ArrayList<_User> m_users; // Store all users with their reviews.
	
	public UserAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) 
			throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		m_users = new ArrayList<_User>();
	}
	
	//Load all the users.
	public void loadUserDir(String folder){
		int count = 0;
		if(folder == null || folder.isEmpty())
			return;
		File dir = new File(folder);
		for(File f: dir.listFiles()){
			if(f.isFile()){
				loadOneUser(f.getAbsolutePath());
				count++;
			} else if (f.isDirectory())
				loadUserDir(f.getAbsolutePath());
		}
		System.out.format("%d users are loaded from %s.", count, folder);
	}
	
	// Load one file as a user here. 
	public void loadOneUser(String filename){
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			String[] names = filename.split("/");
			int endIndex = names[names.length-1].lastIndexOf(".");
			String userID = names[names.length-1].substring(0, endIndex); //UserId is contained in the filename.
			// Skip the first line since it is user name.
			reader.readLine(); 
			
			String reviewID, source, category;
			ArrayList<_Review> reviews = new ArrayList<_Review>();
			_Review review;
			int ylabel;
			long timestamp;
			while((line = reader.readLine()) != null){
				reviewID = line;
				source = reader.readLine(); // review content
				category = reader.readLine(); // review category
				ylabel = Integer.valueOf(reader.readLine());
				timestamp = Long.valueOf(reader.readLine());
				
				// Construct the new review.
				if(ylabel != 3){
					ylabel = (ylabel >= 4) ? 1:0;
					review = new _Review(m_corpus.getCollection().size(), source, ylabel, userID, reviewID, category, timestamp);
					if(AnalyzeDoc(review)) //Create the sparse vector for the review.
						reviews.add(review);
				}
			}
			
			if(reviews.size() != 0){
				m_users.add(new _User(userID, reviews)); //create new user from the file.
				m_corpus.addDocs(reviews);
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}

	//Return all the users.
	public ArrayList<_User> getUsers(){
		return m_users;
	}
}
