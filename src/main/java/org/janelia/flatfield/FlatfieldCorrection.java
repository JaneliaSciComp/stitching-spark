package org.janelia.flatfield;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.flatfield.FlatfieldCorrectionSolver.ModelType;
import org.janelia.flatfield.FlatfieldCorrectionSolver.RegularizerModelType;
import org.janelia.stitching.TileInfo;
import org.janelia.stitching.TileInfoJSONProvider;
import org.janelia.stitching.Utils;
import org.janelia.util.ImageImporter;
import org.kohsuke.args4j.CmdLineException;

import ij.IJ;
import ij.ImagePlus;
import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.img.imageplus.ImagePlusImg;
import net.imglib2.img.imageplus.ImagePlusImgs;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import net.imglib2.view.IntervalView;
import net.imglib2.view.RandomAccessiblePair;
import net.imglib2.view.RandomAccessiblePairNullable;
import net.imglib2.view.Views;
import scala.Tuple2;


public class FlatfieldCorrection implements Serializable, AutoCloseable
{
	private static final long serialVersionUID = -8987192045944606043L;

	private final String histogramsPath, solutionPath;

	private transient final JavaSparkContext sparkContext;
	private transient final TileInfo[] tiles;

	private final long[] fullTileSize;
	private final Interval workingInterval;

	private final FlatfieldCorrectionArguments args;

	public static void main( final String[] args ) throws CmdLineException, IOException
	{
		final FlatfieldCorrectionArguments argsParsed = new FlatfieldCorrectionArguments( args );
		if ( !argsParsed.parsedSuccessfully() )
			System.exit( 1 );

		try ( final FlatfieldCorrection driver = new FlatfieldCorrection( argsParsed ) )
		{
			driver.run();
		}
		System.out.println("Done");
	}


	public static < U extends NativeType< U > & RealType< U > > RandomAccessiblePair< U, U > loadCorrectionImages( final String vPath, final String zPath )
	{
		final ImagePlus vImp = ImageImporter.openImage( vPath );
		final ImagePlus zImp = ImageImporter.openImage( zPath );
		if ( vImp == null || zImp == null )
			return null;

		Utils.workaroundImagePlusNSlices( vImp );
		Utils.workaroundImagePlusNSlices( zImp );

		final ImagePlusImg< U, ? > vImg = ImagePlusImgs.from( vImp );
		final ImagePlusImg< U, ? > zImg = ImagePlusImgs.from( zImp );

		return new RandomAccessiblePair< >(
				vImg,
				zImg.numDimensions() < vImg.numDimensions() ? Views.extendBorder( Views.stack( zImg ) ) : zImg );
	}

	public static <
		T extends NativeType< T > & RealType< T >,
		U extends NativeType< U > & RealType< U > >
	ImagePlusImg< FloatType, ? > applyCorrection( final RandomAccessibleInterval< T > src, final RandomAccessiblePair< U, U > correction )
	{
		final ImagePlusImg< FloatType, ? > dst = ImagePlusImgs.floats( Intervals.dimensionsAsLongArray( src ) );
		final Cursor< T > srcCursor = Views.flatIterable( src ).localizingCursor();
		final Cursor< FloatType > dstCursor = Views.flatIterable( Views.translate( dst, Intervals.minAsLongArray( src ) ) ).cursor();
		final RandomAccessiblePair< U, U >.RandomAccess correctionRandomAccess = correction.randomAccess();
		while ( srcCursor.hasNext() || dstCursor.hasNext() )
		{
			srcCursor.fwd();
			correctionRandomAccess.setPosition( srcCursor );
			dstCursor.next().setReal( srcCursor.get().getRealDouble() * correctionRandomAccess.getA().getRealDouble() + correctionRandomAccess.getB().getRealDouble() );
		}
		return dst;
	}



	public FlatfieldCorrection( final FlatfieldCorrectionArguments args ) throws IOException
	{
		this.args = args;

		tiles = TileInfoJSONProvider.loadTilesConfiguration( args.inputFilePath() );
		fullTileSize = getMinTileSize( tiles );
		workingInterval = args.cropMinMaxInterval( fullTileSize );

		System.out.println( "Working interval is at " + Arrays.toString( Intervals.minAsLongArray( workingInterval ) ) + " of size " + Arrays.toString( Intervals.dimensionsAsLongArray( workingInterval ) ) );

		final String basePath = args.inputFilePath().substring( 0, args.inputFilePath().lastIndexOf( "." ) );
		final String outputPath = basePath + "/" + ( args.cropMinMaxIntervalStr() == null ? "fullsize" : args.cropMinMaxIntervalStr() );
		histogramsPath = basePath + "/" + "histograms";
		solutionPath = outputPath + "/" + "solution";

		// check if all tiles have the same size
		for ( final TileInfo tile : tiles )
			for ( int d = 0; d < tile.numDimensions(); d++ )
				if ( tile.getSize(d) != fullTileSize[ d ] )
				{
					System.out.println("Assumption failed: not all the tiles are of the same size");
					System.exit(1);
				}


		sparkContext = new JavaSparkContext( new SparkConf()
				.setAppName( "IlluminationCorrection3D" )
				//.set( "spark.driver.maxResultSize", "8g" )
				.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
				//.set( "spark.kryoserializer.buffer.max", "2047m" )
				.registerKryoClasses( new Class[] { Short.class, Integer.class, Long.class, Double.class, TreeMap.class, TreeMap[].class, long[].class, short[][].class, double[].class, List.class, Tuple2.class, Interval.class, FinalInterval.class, ArrayImg.class, DoubleType.class, DoubleArray.class } )
				.set( "spark.rdd.compress", "true" )
				//.set( "spark.executor.heartbeatInterval", "10000000" )
				//.set( "spark.network.timeout", "10000000" )
			);
	}

	@Override
	public void close()
	{
		if ( sparkContext != null )
			sparkContext.close();
	}


	public < T extends NativeType< T > & RealType< T >, V extends TreeMap< Short, Integer > >
	void run() throws FileNotFoundException
	{
		long elapsed = System.nanoTime();

		final HistogramsProvider histogramsProvider = new HistogramsProvider(
				sparkContext,
				workingInterval,
				histogramsPath,
				tiles,
				fullTileSize,
				args.histMinValue(), args.histMaxValue(), args.bins() );

		System.out.println( "Working with stack of size " + tiles.length );
		System.out.println( "Output directory: " + solutionPath );

		System.out.println( "Loading histograms.." );
		final JavaPairRDD< Long, long[] > rddFullHistograms = histogramsProvider.getHistograms();

		final long[] referenceHistogram = histogramsProvider.getReferenceHistogram();
		System.out.println( "Obtained reference histogram of size " + referenceHistogram.length );
		System.out.println( "Reference histogram:");
		System.out.println( Arrays.toString( referenceHistogram ) );

		final HistogramSettings histogramSettings = histogramsProvider.getHistogramSettings();

		// Define the transform and calculate the image size on each scale level
		final AffineTransform3D downsamplingTransform = new AffineTransform3D();
		downsamplingTransform.set(
				0.5, 0, 0, -0.5,
				0, 0.5, 0, -0.5,
				0, 0, 0.5, -0.5
			);

		final ShiftedDownsampling shiftedDownsampling = new ShiftedDownsampling( sparkContext, workingInterval, downsamplingTransform );
		final FlatfieldCorrectionSolver solver = new FlatfieldCorrectionSolver( sparkContext );

		final int iterations = 16;
		Pair< RandomAccessibleInterval< DoubleType >, RandomAccessibleInterval< DoubleType > > lastSolution = null;
		for ( int iter = 0; iter < iterations; iter++ )
		{
			Pair< RandomAccessibleInterval< DoubleType >, RandomAccessibleInterval< DoubleType > > downsampledSolution = null;

			// solve in a bottom-up fashion (starting from the smallest scale level)
			for ( int scale = shiftedDownsampling.getNumScales() - 1; scale >= 0; scale-- )
			{
				final Pair< RandomAccessibleInterval< DoubleType >, RandomAccessibleInterval< DoubleType > > solution;

				final ModelType modelType;
				final RegularizerModelType regularizerModelType;

				if ( iter == 0 )
					modelType = scale >= shiftedDownsampling.getNumScales() / 2 ? ModelType.FixedTranslationAffineModel : ModelType.FixedScalingAffineModel;
				else
					modelType = iter % 2 == 1 ? ModelType.FixedTranslationAffineModel : ModelType.FixedScalingAffineModel;

				regularizerModelType = iter == 0 && scale == shiftedDownsampling.getNumScales() - 1 ? RegularizerModelType.IdentityModel : RegularizerModelType.AffineModel;

				try ( ShiftedDownsampling.PixelsMapping pixelsMapping = shiftedDownsampling.new PixelsMapping( scale ) )
				{
					final RandomAccessiblePairNullable< DoubleType, DoubleType > regularizer;
					if ( regularizerModelType == RegularizerModelType.AffineModel )
					{
						final RandomAccessible< DoubleType > scalingRegularizer;
						final RandomAccessible< DoubleType > translationRegularizer;

						if ( modelType != ModelType.FixedScalingAffineModel || lastSolution == null )
							scalingRegularizer = downsampledSolution != null ? shiftedDownsampling.upsample( downsampledSolution.getA(), scale ) : null;
						else
							scalingRegularizer = shiftedDownsampling.downsampleSolutionComponent( lastSolution.getA(), pixelsMapping );

						if ( modelType != ModelType.FixedTranslationAffineModel || lastSolution == null )
							translationRegularizer = downsampledSolution != null ? shiftedDownsampling.upsample( downsampledSolution.getB(), scale ) : null;
						else
							translationRegularizer = shiftedDownsampling.downsampleSolutionComponent( lastSolution.getB(), pixelsMapping );

						regularizer = new RandomAccessiblePairNullable<>( scalingRegularizer, translationRegularizer );
					}
					else
					{
						regularizer = null;
					}

					final JavaPairRDD< Long, long[] > rddDownsampledHistograms = shiftedDownsampling.downsampleHistograms(
							rddFullHistograms,
							pixelsMapping );

					solution = solver.leastSquaresInterpolationFit(
							rddDownsampledHistograms,
							referenceHistogram,
							histogramSettings,
							pixelsMapping,
							regularizer,
							modelType,
							regularizerModelType );
				}

				downsampledSolution = solution;
				saveSolution( iter, scale, solution );
			}

			lastSolution = downsampledSolution;

			if ( iter % 2 == 0 )
			{
				final RandomAccessibleInterval< DoubleType > averageTranslationalComponent = averageSolutionComponent( lastSolution.getB() );
				saveSolutionComponent( iter, 0, averageTranslationalComponent, "z_avg" );

				lastSolution = new ValuePair<>(
						lastSolution.getA(),
						Views.interval( Views.extendBorder( Views.stack( averageTranslationalComponent ) ), lastSolution.getA() ) );

				//saveSolutionComponent( iter, 0, lastSolution.getB(), "z_avg_extended-test" );
			}
		}

		elapsed = System.nanoTime() - elapsed;
		System.out.println( "----------" );
		System.out.println( String.format( "Took %f mins", elapsed / 1e9 / 60 ) );
	}


	@SuppressWarnings("unchecked")
	private RandomAccessibleInterval< DoubleType > averageSolutionComponent( final RandomAccessibleInterval< DoubleType > solutionComponent )
	{
		final RandomAccessibleInterval< DoubleType > dst = ArrayImgs.doubles( new long[] { solutionComponent.dimension( 0 ), solutionComponent.dimension( 1 ) } );

		final IntervalView< DoubleType > src = Views.interval( solutionComponent, new FinalInterval(
				new long[] { dst.min( 0 ), dst.min( 1 ), solutionComponent.min( 2 ) + 3 },
				new long[] { dst.max( 0 ), dst.max( 1 ), solutionComponent.max( 2 ) - 3 } ) );

		for ( long slice = src.min( 2 ); slice <= src.max( 2 ); slice++ )
		{
			final Cursor< DoubleType > srcSliceCursor = Views.flatIterable( Views.hyperSlice( src, 2, slice ) ).cursor();
			final Cursor< DoubleType > dstCursor = Views.flatIterable( dst ).cursor();

			while ( dstCursor.hasNext() || srcSliceCursor.hasNext() )
				dstCursor.next().add( srcSliceCursor.next() );
		}

		final Cursor< DoubleType > dstCursor = Views.iterable( dst ).cursor();
		while ( dstCursor.hasNext() )
		{
			final DoubleType val = dstCursor.next();
			val.set( val.get() / src.dimension( 2 ) );
		}

		return dst;
	}


	private void saveSolution( final int iteration, final int scale, final Pair< RandomAccessibleInterval< DoubleType >, RandomAccessibleInterval< DoubleType > > solution )
	{
		if ( solution.getA() != null )
			saveSolutionComponent( iteration, scale, solution.getA(), "v" );

		if ( solution.getB() != null )
			saveSolutionComponent( iteration, scale, solution.getB(), "z" );
	}

	private void saveSolutionComponent( final int iteration, final int scale, final RandomAccessibleInterval< DoubleType > solutionComponent, final String title )
	{
		final String path = solutionPath + "/iter" + iteration + "/" + scale + "/" + title + ".tif";

		Paths.get( path ).getParent().toFile().mkdirs();

		final ImagePlus imp = ImageJFunctions.wrap( solutionComponent, title );
		Utils.workaroundImagePlusNSlices( imp );
		IJ.saveAsTiff( imp, path );
	}


	private static long[] getMinTileSize( final TileInfo[] tiles )
	{
		final long[] minSize = tiles[ 0 ].getSize().clone();
		for ( final TileInfo tile : tiles )
			for ( int d = 0; d < minSize.length; d++ )
				if (minSize[ d ] > tile.getSize( d ))
					minSize[ d ] = tile.getSize( d );
		return minSize;
	}
}