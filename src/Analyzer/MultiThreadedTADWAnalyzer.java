package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class MultiThreadedTADWAnalyzer extends MultiThreadedUserAnalyzer {

    HashSet<String> m_userIds = new HashSet<String>();
    HashMap<String, HashSet<String>> m_userTrainDocsMap = new HashMap<>();

    public MultiThreadedTADWAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, int numberOfCores, boolean b)
					throws InvalidFormatException, FileNotFoundException, IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b);
    }

    // Load one file as a user here, treat one user as one document for later tf-idf calculation
    public void loadUserTxt(String filename){
        try {
            System.out.println("Start loading users.....");
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String userId, sources;
            int count = 0;

            while((userId = reader.readLine()) != null){
                userId = userId.trim();
                sources = reader.readLine();
                count++;
                if(count % 1000 == 0)
                    System.out.print(".");
                ArrayList<_Review> reviews = new ArrayList<>();
                if(sources.length() > 0) {
                    _Review review = new _Review(-1, sources, 0, userId, "", "", 0);
                    if(AnalyzeDoc(review)){
                        reviews.add(review);
                        m_users.add(new _User(userId, m_classNo, reviews)); //create new user from the file.
                    } else{
                        System.out.println("The user does not have valid document!");
                    }
                }
            }
            reader.close();
            System.out.format("[Info]\n%d/%d users are loaded!!!\n", m_users.size(), count);
        } catch(IOException e){
            e.printStackTrace();
        }
    }


    public void printData4TADW(String filename){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            writer.format("%d\t%d\n", m_users.size(), m_featureNames.size());
            for(_User user: m_users) {
                String uid = user.getUserID();
                if(user.getReviews().size() > 0){
                    if(user.getReviews().size() != 1)
                        System.out.println("[error]The user has <1 reviews!!!");
                    _SparseFeature[] fvs = user.getReviews().get(0).getSparse();
                    writer.write(uid+"\t");
                    for(_SparseFeature fv: fvs){
                        writer.write(String.format("%d\t%.3f\t", fv.getIndex(), fv.getValue()));
                    }
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("Finish writing %d users' data for TADW.\n", m_users.size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }

}
