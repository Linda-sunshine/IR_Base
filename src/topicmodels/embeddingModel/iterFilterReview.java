package topicmodels.embeddingModel;

import json.JSONArray;
import json.JSONObject;

import java.io.*;
import java.util.*;

/**
 * @author Lu Lin
 */
public class iterFilterReview {

    public HashMap<String, int[]> userIDMap;
    public HashMap<String, int[]> itemIDMap;

    public iterFilterReview(){
        this.userIDMap = new HashMap<String, int[]>();
        this.itemIDMap = new HashMap<String, int[]>();
    }

    //load data from json file to model, object = {user, item, review}
    public void loadData(String fileName, int threshold, String object){

        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
            StringBuffer buffer = new StringBuffer(1024);
            String line;

            int userNum = 0, itemNum = 0;
            while((line=reader.readLine())!=null) {
                JSONObject obj = new JSONObject(line.toString());

                if(object == "user" && obj.has("user_id")){
                    String userID = obj.getString("user_id");
                    int review_count = obj.getInt("review_count");
                    if (review_count >= threshold) {
                        int[] value = {userNum++, 0};
                        userIDMap.put(userID, value);
                    }
                }

                if(object == "item" && obj.has("business_id")){
                    String itemID = obj.getString("business_id");
                    int review_count = obj.getInt("review_count");
                    if(review_count >= threshold){
                        int[] value = {itemNum++, 0};
                        itemIDMap.put(itemID, value);
                    }
                }

                if(object == "review" && obj.has("review_id")){
                    String reviewID = obj.getString("review_id");
                    String userID = obj.getString("user_id");
                    String itemID = obj.getString("business_id");
                    if(userIDMap.containsKey(userID) && itemIDMap.containsKey(itemID)){
                        int[] uValue = {userIDMap.get(userID)[0], userIDMap.get(userID)[1] + 1};
                        int[] iValue = {itemIDMap.get(itemID)[0], itemIDMap.get(itemID)[1] + 1};
                        userIDMap.put(userID, uValue);
                        itemIDMap.put(itemID, iValue);
                    }
                }
            }
            reader.close();
        } catch (Exception e) {
            System.out.print("! FAIL to load " + object + " json file...");
        }
    }

    public void iterFiltering(String userFileName, String itemFileName, String reviewFileName,
                              int userThreshold, int itemThreshold, int filterIterNum){

        //load user and item data with review_count no less than threshold
        userIDMap = new HashMap<String, int[]>();
        loadData(userFileName, userThreshold, "user");
        itemIDMap = new HashMap<String, int[]>();
        loadData(itemFileName, itemThreshold, "item");

        System.out.println("Loading data finished: " + userIDMap.size()
                + " users; " + itemIDMap.size() + " items.");

        loadData(reviewFileName, 1, "review");


        //iteratively filter user/item with too few reviews to get a dense bipartite
        int i;
        for(i = 0; i < filterIterNum; i++){
            int userNum = userIDMap.size();
            int itemNum = itemIDMap.size();

            //delete user with too few reviews by setting all its row elements to be 0
            Iterator<Map.Entry<String, int[]>> it = userIDMap.entrySet().iterator();
            while(it.hasNext()){
                Map.Entry<String, int[]> entry = it.next();
                if(entry.getValue()[1] < userThreshold){
                    it.remove();
                }
            }

            //delete item with too few reviews by setting all its column elements to be 0
            it = itemIDMap.entrySet().iterator();
            while(it.hasNext()){
                Map.Entry<String, int[]> entry = it.next();
                if(entry.getValue()[1] < itemThreshold){
                    it.remove();
                }
            }
            System.out.println("-- after " + i + " iterations: remain "
                    + userIDMap.size() + " users; " + itemIDMap.size() + " items.");

            // renew their count by parse review.json
            for(Map.Entry<String, int[]> entry: userIDMap.entrySet()){
                int[] value = {entry.getValue()[0], 0};
                entry.setValue(value);
            }
            for(Map.Entry<String, int[]> entry: itemIDMap.entrySet()){
                int[] value = {entry.getValue()[0], 0};
                entry.setValue(value);
            }
            loadData(reviewFileName,1 ,"review");
        }

        System.out.println("Filtering finished: after " + i + " iterations");

        //save dense sub bipartite graph
        System.out.println("-- contains: " + userIDMap.size() + " users; " + itemIDMap.size() + " items.");
    }

    public void saveFilterdData(String inFileName, String outFileName){

        try {
            FileWriter file = new FileWriter(outFileName);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(inFileName), "UTF-8"));
            StringBuffer buffer = new StringBuffer(1024);
            String line;

            int userNum = 0, itemNum = 0;
            while ((line = reader.readLine()) != null) {
                JSONObject obj = new JSONObject(line.toString());

                if(obj.has("review_id")) {
                    String userID = obj.getString("user_id");
                    String itemID = obj.getString("business_id");
                    if (userIDMap.containsKey(userID) && itemIDMap.containsKey(itemID)) {
                        file.write(obj.toString());
                        file.write('\n');
                    }
                }
            }
            file.flush();
            file.close();
        }catch(Exception e) {
            System.err.println("! FAIL to load review json file or open new file...");//fail to parse a json document
        }
    }

    public void clusterDataByItem(String inFileName, String outFileName){
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(inFileName), "UTF-8"));
            StringBuffer buffer = new StringBuffer(1024);
            String line;
            HashMap<String, Integer> itemMap = new HashMap<String, Integer>();

            //sort item by review count
            while ((line = reader.readLine()) != null) {
                JSONObject obj = new JSONObject(line.toString());

                if(obj.has("review_id")) {
                    String itemID = obj.getString("business_id");
                    itemMap.put(itemID, 0);
                }
            }
            reader.close();
            System.out.println("load " + itemMap.size() + " items.");

            reader = new BufferedReader(new InputStreamReader(new FileInputStream(inFileName), "UTF-8"));
            buffer = new StringBuffer(1024);
            while((line = reader.readLine()) != null){
                JSONObject obj = new JSONObject(line.toString());

                if(obj.has("review_id")){
                    String itemID = obj.getString("business_id");
                    itemMap.put(itemID, itemMap.get(itemID)+1);
                }
            }
            reader.close();

            Object[] temp = itemMap.entrySet().toArray();
            Arrays.sort(temp, new Comparator() {
                public int compare(Object o1, Object o2) {
                    return ((Map.Entry<String, Integer>) o2).getValue()
                            .compareTo(((Map.Entry<String, Integer>) o1).getValue());
                }
            });

            for (int i = 0; i < 5; i++) {
                Object item = temp[i];
                String itemID = ((Map.Entry<String, Integer>) item).getKey();
                int reviewCount = ((Map.Entry<String, Integer>) item).getValue();
                FileWriter file = new FileWriter(outFileName + reviewCount + "_" + itemID + ".json");

                reader = new BufferedReader(new InputStreamReader(new FileInputStream(inFileName), "UTF-8"));
                buffer = new StringBuffer(1024);
                JSONArray jarry = new JSONArray();
                while((line = reader.readLine()) != null){
                    JSONObject obj = new JSONObject(line.toString());

                    if(obj.has("review_id")){
                        String iID = obj.getString("business_id");

                        if(iID.equals(itemID)){
                            jarry.put(obj);
                        }
                    }
                }
                JSONObject thisItem = new JSONObject();
                thisItem.put("ProductID", itemID);
                thisItem.put("reviews", jarry);
                file.write(thisItem.toString());
                file.flush();
                file.close();
                System.out.println(i + " item has: " + reviewCount + "reviews.");
            }


        }catch(Exception e) {
            System.err.println("! FAIL to load review json file or open new file...");//fail to parse a json document
        }
    }

    public void clusterDataByUser(String inFileName, String outFileName){
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(inFileName), "UTF-8"));
            StringBuffer buffer = new StringBuffer(1024);
            String line;
            HashMap<String, Integer> userMap = new HashMap<String, Integer>();

            //sort item by review count
            while ((line = reader.readLine()) != null) {
                JSONObject obj = new JSONObject(line.toString());

                if(obj.has("review_id")) {
                    String userID = obj.getString("user_id");
                    userMap.put(userID, 0);
                }
            }
            reader.close();
            System.out.println("load " + userMap.size() + " users.");

            reader = new BufferedReader(new InputStreamReader(new FileInputStream(inFileName), "UTF-8"));
            buffer = new StringBuffer(1024);
            while((line = reader.readLine()) != null){
                JSONObject obj = new JSONObject(line.toString());

                if(obj.has("review_id")){
                    String userID = obj.getString("user_id");
                    userMap.put(userID, userMap.get(userID)+1);
                }
            }
            reader.close();

            Object[] temp = userMap.entrySet().toArray();
            Arrays.sort(temp, new Comparator() {
                public int compare(Object o1, Object o2) {
                    return ((Map.Entry<String, Integer>) o2).getValue()
                            .compareTo(((Map.Entry<String, Integer>) o1).getValue());
                }
            });

            for (int i = 0; i < temp.length; i++) {
                Object user = temp[i];
                String userID = ((Map.Entry<String, Integer>) user).getKey();
                int reviewCount = ((Map.Entry<String, Integer>) user).getValue();
                if(reviewCount > 50){
                    continue;
                }

                FileWriter file = new FileWriter(outFileName + reviewCount + "_" + userID + ".json");

                reader = new BufferedReader(new InputStreamReader(new FileInputStream(inFileName), "UTF-8"));
                buffer = new StringBuffer(1024);
                JSONArray jarry = new JSONArray();
                while((line = reader.readLine()) != null){
                    JSONObject obj = new JSONObject(line.toString());

                    if(obj.has("review_id")){
                        String iID = obj.getString("user_id");

                        if(iID.equals(userID)){
                            jarry.put(obj);
                        }
                    }
                }
                JSONObject thisItem = new JSONObject();
                thisItem.put("userID", userID);
                thisItem.put("reviews", jarry);
                file.write(thisItem.toString());
                file.flush();
                file.close();
                System.out.println(i + " user has: " + reviewCount + "reviews.");
                break;
            }


        }catch(Exception e) {
            System.err.println("! FAIL to load review json file or open new file...");//fail to parse a json document
        }
    }

    public static void main(String[] args) throws FileNotFoundException {

        int userMinCount = 20;
        int itemMinCount = 50;
        int filtIterNum = 25;

        String itemFileName = "./myData/business.json";
        String userFileName = "./myData/user.json";
        String reviewFileName = "./myData/review.json";
        String denseFileName = "./myData/denseReview_" + userMinCount + "_" + itemMinCount + ".json";

        iterFilterReview preprocessor = new iterFilterReview();
//        preprocessor.iterFiltering(userFileName, itemFileName, reviewFileName,
//                userMinCount, itemMinCount, filtIterNum);
//        preprocessor.saveFilterdData(reviewFileName, denseFileName);

        preprocessor.clusterDataByUser(denseFileName, "./myData/byUser/");
//        etbirModel.readData(dataFileName);
//        etbirModel.readVocabulary(vocFileName);
//        etbirModel.EM();
    }
}
