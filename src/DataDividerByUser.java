import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;

/* Preprocess raw data and merge the data with the same userID together */
public class DataDividerByUser {
	public static class DataDividerMapper extends Mapper<LongWritable, Text, IntWritable, Text> {

		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

			/*
			* Input: userID, movieID, rating
			* Output key: userID
			* Output value: movieID + rating
			* */
			String [] user_movie_rating = value.toString().trim().split(",");
			String outputKey = user_movie_rating[0];

			/* outputValue = movieID:rating */
			String outputValue = user_movie_rating[1] + ":" + user_movie_rating[2];
			context.write(new IntWritable(Integer.parseInt(outputKey)), new Text(outputValue));
			//divide data by user
		}
	}

	public static class DataDividerReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
		/* Merge data with the same userID */
		@Override
		public void reduce(IntWritable key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			/* Inputkey: userID
			* Input Value: <movie:rating,...>
			* Outputkey: userID
			* OutputValue: movie1:rating1, movie2:rating2
			* */
			StringBuilder sb = new StringBuilder();
			for (Text value:values) {
				sb.append("," + value.toString());
			}

			/* remove the first comma */
			context.write(key, new Text(sb.toString().replaceFirst(",", "")));

			/*
			* user1 \t movie1:3.5, movie2:4 ...
			* */

		}
	}

	public static void main(String[] args) throws Exception {

		Configuration conf = new Configuration();

		Job job = Job.getInstance(conf);
		job.setMapperClass(DataDividerMapper.class);
		job.setReducerClass(DataDividerReducer.class);

		job.setJarByClass(DataDividerByUser.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(Text.class);

		TextInputFormat.setInputPaths(job, new Path(args[0]));
		TextOutputFormat.setOutputPath(job, new Path(args[1]));

		job.waitForCompletion(true);
	}

}
