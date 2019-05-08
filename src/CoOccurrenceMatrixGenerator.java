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

public class CoOccurrenceMatrixGenerator {
	public static class MatrixGeneratorMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

		/* Build coocurrence matrix */
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			/* Input value: userid \t movie1: rating, movie2: rating...
			   Output key: movie1:movie2
			   Output value: 1
			Calculate each user rating list: <movieA, movieB>
			*/
			String[] user_movies = value.toString().trim().split("\t");
			/* if there is only userID and no ratings */
			if (user_movies.length < 2) {
				/* add log to record userIDs without any ratings */
				return;
			}

			String[] movies = user_movies[1].split(",");
			for (int i = 0; i < movies.length; i++) {
				/* movie[i] = movieID:rating */
				String movieA = movies[i].trim().split(";")[0];
				for (int j = 0; j < movies.length; j++) {
					String movieB = movies[j].trim().split(":")[0];
					String outputKey = movieA + ":" + movieB;
					context.write(new Text(outputKey), new IntWritable(1));
				}
			}


		}
	}

	public static class MatrixGeneratorReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			/*
			Inputkey: movieA:movieB
			InputValue: value = iterable<1, 1, 1>
			Calculate each two movies have been rated by how many users
			*/
			int sum = 0;
			for (IntWritable value:values) {
				sum += value.get();
			}
			context.write(key, new IntWritable(sum));
		}
	}
	
	public static void main(String[] args) throws Exception{
		
		Configuration conf = new Configuration();
		
		Job job = Job.getInstance(conf);
		job.setMapperClass(MatrixGeneratorMapper.class);
		job.setReducerClass(MatrixGeneratorReducer.class);
		
		job.setJarByClass(CoOccurrenceMatrixGenerator.class);
		
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		
		TextInputFormat.setInputPaths(job, new Path(args[0]));
		TextOutputFormat.setOutputPath(job, new Path(args[1]));
		
		job.waitForCompletion(true);
		
	}
}
