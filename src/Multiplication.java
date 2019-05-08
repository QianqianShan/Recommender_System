import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.chain.ChainMapper;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/* Multiply co-occurrence matrix with user rating matrix */
public class Multiplication {
	public static class CooccurrenceMapper extends Mapper<LongWritable, Text, Text, Text> {

		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			/*
			  Input: movieB \t movieA=relation
			  Outputkey: movieB
			  Outputvalue: movieA = relativeRelation
			*/

			String[] movie_relation = value.toString().trim().split("\t");
			context.write(new Text(movie_relation[0]), new Text(movie_relation[1]));
		}
	}

	public static class RatingMapper extends Mapper<LongWritable, Text, Text, Text> {

		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

			/*
			  Input: userID, movieID, rating
			  OutputKey: movieID
			  OutputValue: userID:rating
			 */
			String[] user_movie_rating = value.toString().trim().split(",");
			context.write(new Text(user_movie_rating[1]), new Text(user_movie_rating[0] + ":" + user_movie_rating[2]));
		}
	}

	public static class MultiplicationReducer extends Reducer<Text, Text, Text, DoubleWritable> {
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {

			/*
			  1. Count data from two mappers and store them into two Maps for later use
			  2. Element-wise multiplication of co-occurence matrix and user ratings
			  3. Write data

			  Inputkey = movieB
			  Inputvalue = <movieA=relation, movieC=relation... userA:rating, userB:rating...>
			  Outputkey: movieA+":" + userID
			  Outputvalue: relation * rating
			Collect the data for each movie, then do the multiplication
			*/

			Map<String, Double> relationMap = new HashMap<String, Double>();
			Map<String, Double> ratingmap = new HashMap<String, Double>();

			/* Process relation and rating inputvalues separately */
			for (Text value:values) {
				if (value.toString().contains("=")) {
					/* value: movieA = relation */
					String[] movieA_relation = value.toString().split("=");
					relationMap.put(movieA_relation[0], Double.parseDouble(movieA_relation[1]));
				} else {
					String[] user_rating = value.toString().split(":");
					ratingMap.put(user_rating[0], Double.parseDouble(user_rating[1]));
				}
			}
            /* Multiplication */
			for (Map.Entry<String, Double> entry:relationMap.entrySet()) {
				/* co-occurence info from relationMap */
				String movieA = entry.getKey();
				double relation = entry.getValue();
                /* Rating info from ratingMap */
				for (Map.Entry<String, Double> element:ratingMap.entrySet()) {
					String userID = element.getKey();
					double rating = element.getValue();
					double outputValue = relation * rating;
					String outputKey = userID + ":" + movieA;
					context.write(new Text(outputKey), new DoubleWritable(outputValue));
					/* output example: user1:movie23 \t 0.25 * 8.5 */
				}
			}
		}
	}


	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();

		Job job = Job.getInstance(conf);
		job.setJarByClass(Multiplication.class);

		ChainMapper.addMapper(job, CooccurrenceMapper.class, LongWritable.class, Text.class, Text.class, Text.class, conf);
		ChainMapper.addMapper(job, RatingMapper.class, Text.class, Text.class, Text.class, Text.class, conf);

		job.setMapperClass(CooccurrenceMapper.class);
		job.setMapperClass(RatingMapper.class);

		job.setReducerClass(MultiplicationReducer.class);

		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(DoubleWritable.class);

		MultipleInputs.addInputPath(job, new Path(args[0]), TextInputFormat.class, CooccurrenceMapper.class);
		MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, RatingMapper.class);

		TextOutputFormat.setOutputPath(job, new Path(args[2]));
		
		job.waitForCompletion(true);
	}
}
