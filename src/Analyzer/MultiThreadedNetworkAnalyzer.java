package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._Review;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The analyzer aims at the directional network information.
 */
public class MultiThreadedNetworkAnalyzer extends MultiThreadedLinkPredAnalyzer {

    public MultiThreadedNetworkAnalyzer(String tokenModel, int classNo,
                                        String providedCV, int Ngram, int threshold, int numberOfCores, boolean b)
            throws InvalidFormatException, FileNotFoundException, IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b);
    }

    // save the network for later use
    public void saveNetwork(String filename, HashMap<String, HashSet<String>> map){
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: map.keySet()){
                writer.write(uid + '\t');
                for(String it: map.get(uid)){
                    writer.write(it + '\t');
                }
                writer.write('\n');
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // load the interactions, filter the users who are not in the user set
    public void loadInteractions(String filename){
        try{
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            // load the interactions first
            while((line = reader.readLine()) != null){
                String[] users = line.trim().split("\t");
                String[] strs = Arrays.copyOfRange(users, 1, users.length);
                String[] interactions = filterNonExistInteractions(strs);
                if(!m_userIDIndex.containsKey(users[0]))
                    System.err.println("[error] The user does not exist: " + users[0]);
                else {
                    int uIndex = m_userIDIndex.get(users[0]);
                    m_users.get(uIndex).setFriends(interactions);
                }
            }
            reader.close();

        } catch(IOException e){
            e.printStackTrace();
        }
    }

    protected String[] filterNonExistInteractions(String[] strs){
        if(strs.length <= 1)
            return null;
        ArrayList<String> interactions = new ArrayList<>();
        for(int i=1; i<strs.length; i++){
            if(m_userIDIndex.containsKey(strs[i])){
                interactions.add(strs[i]);
            }
        }
        return interactions.toArray(new String[interactions.size()]);
    }

    // shuffle the whole corpus and save the index information for later use
    public void saveCVIndex(int k, String filename){
        m_corpus.setMasks();
        m_corpus.shuffle(k);
        int[] masks = m_corpus.getMasks();
        ArrayList<_Doc> docs = m_corpus.getCollection();
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int i=0; i<docs.size(); i++){
                _Review r = (_Review) docs.get(i);
                writer.write(String.format("%s,%d,%d\n", r.getUserID(), r.getID(), masks[i]));
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    @Override
    public void constructUserIDIndex(){
        m_userIDIndex = new HashMap<String, Integer>();
        for(int i=0; i<m_users.size(); i++){
            m_userIDIndex.put(m_users.get(i).getUserID(), i);
        }
    }

    // load cv index for all the documents
    public void loadCVIndex(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while((line = reader.readLine()) != null) {
                String[] strs = line.trim().split(",");
                String uId = strs[0];
                int id = Integer.valueOf(strs[1]);
                int mask = Integer.valueOf(strs[2]);

                if(!m_userIDIndex.containsKey(uId))
                    System.out.println("No such user!");
                else {
                    int uIndex = m_userIDIndex.get(uId);
                    if (uIndex > m_users.size())
                        System.out.println("Exceeds the array size!");
                    else {
                        m_users.get(m_userIDIndex.get(uId)).getReviewByID(id).setMask4CV(mask);
                    }
                }
            }
        } catch(IOException e){
            e.printStackTrace();
        }
    }

}
