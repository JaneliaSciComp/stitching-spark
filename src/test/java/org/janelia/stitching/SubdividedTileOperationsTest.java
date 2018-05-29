package org.janelia.stitching;

import static org.junit.Assert.fail;

import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.junit.Assert;
import org.junit.Test;

import net.imglib2.Interval;
import net.imglib2.util.IntervalIndexer;
import net.imglib2.util.Intervals;
import net.imglib2.util.IntervalsHelper;
import net.imglib2.util.IntervalsNullable;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;

public class SubdividedTileOperationsTest
{
	private static final double EPSILON = 1e-9;

	@Test
	public void splitTilesIntoBoxesTest()
	{
		final TileInfo tile = new TileInfo( 3 );
		tile.setStagePosition( new double[] { 100, 200, 300 } );
		tile.setSize( new long[] { 50, 60, 70 } );

		final List< SubdividedTileBox > tileBoxes = SubdividedTileOperations.subdivideTiles( new TileInfo[] { tile }, new int[] { 2, 2, 2 } );
		Assert.assertEquals( 8, tileBoxes.size() );

		// test mins
		final Map< Long, long[] > actualMins = new TreeMap<>();
		for ( final SubdividedTileBox tileBox : tileBoxes )
		{
			final long[] mins = Intervals.minAsLongArray( IntervalsHelper.roundRealInterval( tileBox ) );
			final long key = IntervalIndexer.positionToIndex( mins, tile.getSize() );
			Assert.assertNull( actualMins.put( key, mins ) );
		}
		final long[][] expectedMinsArray = new long[][] {
			new long[] { 0, 0, 0 },
			new long[] { 25, 0, 0 },
			new long[] { 0, 30, 0 },
			new long[] { 25, 30, 0 },
			new long[] { 0, 0, 35 },
			new long[] { 25, 0, 35 },
			new long[] { 0, 30, 35 },
			new long[] { 25, 30, 35 },
		};
		for ( final long[] expectedMin : expectedMinsArray )
		{
			final long key = IntervalIndexer.positionToIndex( expectedMin, tile.getSize() );
			Assert.assertArrayEquals( expectedMin, actualMins.get( key ) );
		}

		// test dimensions
		for ( final SubdividedTileBox tileBox : tileBoxes )
			Assert.assertArrayEquals( new long[] { 25, 30, 35 }, Intervals.dimensionsAsLongArray( IntervalsHelper.roundRealInterval( tileBox ) ) );

		// check that reference to the original tile is preserved
		for ( final SubdividedTileBox tileBox : tileBoxes )
			Assert.assertEquals( tile, tileBox.getFullTile() );
	}

	@Test
	public void transformedMovingTileBoxMiddlePointTest()
	{
		final TileInfo[] tiles = new TileInfo[ 2 ];
		tiles[ 0 ] = new TileInfo( 3 );
		tiles[ 0 ].setIndex( 0 );
		tiles[ 0 ].setStagePosition( new double[] { 100, 200, 300 } );
		tiles[ 0 ].setSize( new long[] { 50, 60, 70 } );

		tiles[ 1 ] = new TileInfo( 3 );
		tiles[ 1 ].setIndex( 1 );
		tiles[ 1 ].setStagePosition( new double[] { 140, 210, 290 } );
		tiles[ 1 ].setSize( new long[] { 90, 80, 70 } );

		final List< SubdividedTileBox > tileBoxes = SubdividedTileOperations.subdivideTiles( tiles, new int[] { 2, 2, 2 } );
		Assert.assertEquals( 16, tileBoxes.size() );

		// test left-top-front tile box of the second tile
		{
			final SubdividedTileBox movingTileBox = tileBoxes.get( 8 );
			Assert.assertArrayEquals( new double[] { 22.5, 20, 17.5 }, SubdividedTileOperations.getTileBoxMiddlePoint( movingTileBox ), EPSILON );
			final Pair< Interval, Interval > transformedInGlobalSpace = TransformedTileOperations.transformTileBoxPair( new SubdividedTileBoxPair( tileBoxes.get( 0 ), movingTileBox ) );
			final Interval transformedSecondTileBox = SubdividedTileOperations.globalToFixedBoxSpace( transformedInGlobalSpace ).getB();
			Assert.assertArrayEquals( new long[] { 40, 10, -10 }, Intervals.minAsLongArray( transformedSecondTileBox ) );
			Assert.assertArrayEquals( movingTileBox.getSize(), Intervals.dimensionsAsLongArray( transformedSecondTileBox ) );
		}

		// test right-bottom-back tile box of the second tile
		{
			final SubdividedTileBox movingTileBox = tileBoxes.get( 15 );
			Assert.assertArrayEquals( new double[] { 67.5, 60, 52.5 }, SubdividedTileOperations.getTileBoxMiddlePoint( movingTileBox ), EPSILON );
			final Pair< Interval, Interval > transformedInGlobalSpace = TransformedTileOperations.transformTileBoxPair( new SubdividedTileBoxPair( tileBoxes.get( 0 ), movingTileBox ) );
			final Interval transformedSecondTileBox = SubdividedTileOperations.globalToFixedBoxSpace( transformedInGlobalSpace ).getB();
			Assert.assertArrayEquals( new long[] { 85, 50, 25 }, Intervals.minAsLongArray( transformedSecondTileBox ) );
			Assert.assertArrayEquals( movingTileBox.getSize(), Intervals.dimensionsAsLongArray( transformedSecondTileBox ) );
		}
	}

	@Test
	public void testOverlappingTileBoxes()
	{
		final TileInfo[] tiles = new TileInfo[ 2 ];
		tiles[ 0 ] = new TileInfo( 3 );
		tiles[ 0 ].setIndex( 0 );
		tiles[ 0 ].setStagePosition( new double[] { 100, 200, 300 } );
		tiles[ 0 ].setSize( new long[] { 50, 60, 70 } );

		tiles[ 1 ] = new TileInfo( 3 );
		tiles[ 1 ].setIndex( 1 );
		tiles[ 1 ].setStagePosition( new double[] { 140, 210, 290 } );
		tiles[ 1 ].setSize( new long[] { 90, 80, 70 } );

		final List< SubdividedTileBox > tileBoxes = SubdividedTileOperations.subdivideTiles( tiles, new int[] { 2, 2, 2 } );
		Assert.assertEquals( 16, tileBoxes.size() );

		final List< SubdividedTileBoxPair > overlappingTileBoxes = SubdividedTileOperations.findOverlappingTileBoxes( tileBoxes, true );
		Assert.assertEquals( 4, overlappingTileBoxes.size() );

		// test references to the original tiles and their order
		for ( final SubdividedTileBoxPair tileBoxPair : overlappingTileBoxes )
		{
			Assert.assertEquals( tiles[ 0 ], tileBoxPair.getA().getFullTile() );
			Assert.assertEquals( tiles[ 1 ], tileBoxPair.getB().getFullTile() );
		}

		// test mins of overlapping pairs
		final Map< Long, SubdividedTileBoxPair > actualMinsFirstTile = new TreeMap<>();
		for ( final SubdividedTileBoxPair tileBoxPair : overlappingTileBoxes )
		{
			final long[] minsFirstTile = Intervals.minAsLongArray( IntervalsHelper.roundRealInterval( tileBoxPair.getA() ) );
			final long key = IntervalIndexer.positionToIndex( minsFirstTile, tiles[ 0 ].getSize() );
			Assert.assertNull( actualMinsFirstTile.put( key, tileBoxPair ) );
		}
		final long[][] expectedMinsFirstTileArray = new long[][] {
			new long[] { 25, 0, 0 },
			new long[] { 25, 30, 0 },
			new long[] { 25, 0, 35 },
			new long[] { 25, 30, 35 },
		};
		final long[][] expectedMinsSecondTileArray = new long[][] {
			new long[] { 0, 0, 0 },
			new long[] { 0, 0, 0 },
			new long[] { 0, 0, 35 },
			new long[] { 0, 0, 35 },
		};
		for ( int i = 0; i < overlappingTileBoxes.size(); ++i )
		{
			final long key = IntervalIndexer.positionToIndex( expectedMinsFirstTileArray[ i ], tiles[ 0 ].getSize() );
			final SubdividedTileBoxPair tileBoxPair = actualMinsFirstTile.get( key );
			Assert.assertArrayEquals( expectedMinsFirstTileArray[ i ], Intervals.minAsLongArray( IntervalsHelper.roundRealInterval( tileBoxPair.getA() ) ) );
			Assert.assertArrayEquals( expectedMinsSecondTileArray[ i ], Intervals.minAsLongArray( IntervalsHelper.roundRealInterval( tileBoxPair.getB() ) ) );
		}
	}

	@Test
	public void testOverlaps() throws PipelineExecutionException
	{
		final TileInfo fixedTile = new TileInfo( 2 );
		fixedTile.setIndex( 0 );
		fixedTile.setStagePosition( new double[] { 50, 50 } );
		fixedTile.setSize( new long[] { 20, 20 } );

		final TileInfo movingTile = new TileInfo( 2 );
		movingTile.setIndex( 1 );
		movingTile.setStagePosition( new double[] { 46, 32 } );
		movingTile.setSize( new long[] { 20, 20 } );

		final List< SubdividedTileBox > tileBoxes = SubdividedTileOperations.subdivideTiles( new TileInfo[] { fixedTile,  movingTile }, new int[] { 2, 2 } );
		Assert.assertEquals( 8, tileBoxes.size() );

		final SubdividedTileBoxPair tileBoxPair;
		{
			SubdividedTileBox fixedTileBox = null;
			SubdividedTileBox movingTileBox = null;
			for ( final SubdividedTileBox tileBox : tileBoxes )
			{
				if ( tileBox.getFullTile().getIndex().intValue() == 0 )
				{
					// top-right box of the fixed tile
					if ( Math.round( tileBox.getPosition( 0 ) ) == 10 && Math.round( tileBox.getPosition( 1 ) ) == 0 )
					{
						if ( fixedTileBox != null )
							fail();
						fixedTileBox = tileBox;
					}
				}
				else if ( tileBox.getFullTile().getIndex().intValue() == 1 )
				{
					// bottom-right box of the moving tile
					if ( Math.round( tileBox.getPosition( 0 ) ) == 10 && Math.round( tileBox.getPosition( 1 ) ) == 10 )
					{
						if ( movingTileBox != null )
							fail();
						movingTileBox = tileBox;
					}
				}
				else
				{
					fail();
				}
			}
			if ( fixedTileBox == null || movingTileBox == null )
				fail();
			tileBoxPair = new SubdividedTileBoxPair( fixedTileBox, movingTileBox );
		}
		Assert.assertArrayEquals( new long[] { 10, 0 }, Intervals.minAsLongArray( IntervalsHelper.roundRealInterval( tileBoxPair.getA() ) ) );
		Assert.assertArrayEquals( new long[] { 10, 10 }, Intervals.minAsLongArray( IntervalsHelper.roundRealInterval( tileBoxPair.getB() ) ) );

		final Pair< Interval, Interval > intervalsInGlobalSpace = new ValuePair<>(
				IntervalsHelper.translate(
						IntervalsHelper.roundRealInterval( tileBoxPair.getA() ),
						Intervals.minAsLongArray( IntervalsHelper.roundRealInterval( tileBoxPair.getA().getFullTile().getStageInterval() ) )
					),
				IntervalsHelper.translate(
						IntervalsHelper.roundRealInterval( tileBoxPair.getB() ),
						Intervals.minAsLongArray( IntervalsHelper.roundRealInterval( tileBoxPair.getB().getFullTile().getStageInterval() ) )
					)
			);

		final Pair< Interval, Interval > intervalsInFixedBoxSpace = SubdividedTileOperations.globalToFixedBoxSpace( intervalsInGlobalSpace );

		Assert.assertArrayEquals( new long[] { 60, 50 }, Intervals.minAsLongArray( IntervalsNullable.intersect( intervalsInGlobalSpace.getA(), intervalsInGlobalSpace.getB() ) ) );
		Assert.assertArrayEquals( new long[] { 65, 51 }, Intervals.maxAsLongArray( IntervalsNullable.intersect( intervalsInGlobalSpace.getA(), intervalsInGlobalSpace.getB() ) ) );

		Assert.assertArrayEquals( new long[] { 0, 0 }, Intervals.minAsLongArray( IntervalsNullable.intersect( intervalsInFixedBoxSpace.getA(), intervalsInFixedBoxSpace.getB() ) ) );
		Assert.assertArrayEquals( new long[] { 5, 1 }, Intervals.maxAsLongArray( IntervalsNullable.intersect( intervalsInFixedBoxSpace.getA(), intervalsInFixedBoxSpace.getB() ) ) );

		Assert.assertArrayEquals( new long[] { 0, 0 }, Intervals.minAsLongArray( intervalsInFixedBoxSpace.getA() ) );
		Assert.assertArrayEquals( new long[] { -4, -8 }, Intervals.minAsLongArray( intervalsInFixedBoxSpace.getB() ) );

		final double ellipseRadius = 2;
		final double[] offsetsMeanValues = Intervals.minAsDoubleArray( intervalsInFixedBoxSpace.getB() );
		final double[][] offsetsCovarianceMatrix = new double[][] { new double[] { 1, 0, 0 }, new double[] { 0, 1, 0 }, new double[] { 0, 0, 1 } };
		final ErrorEllipse errorEllipse = new ErrorEllipse( ellipseRadius, offsetsMeanValues, offsetsCovarianceMatrix );

		final Pair< Interval, Interval > adjustedOverlaps = SubdividedTileOperations.getAdjustedOverlapIntervals( intervalsInFixedBoxSpace, errorEllipse );

		Assert.assertArrayEquals( new long[] { 0, 0 }, Intervals.minAsLongArray( adjustedOverlaps.getA() ) );
		Assert.assertArrayEquals( new long[] { 7, 3 }, Intervals.maxAsLongArray( adjustedOverlaps.getA() ) );

		Assert.assertArrayEquals( new long[] { 2, 6 }, Intervals.minAsLongArray( adjustedOverlaps.getB() ) );
		Assert.assertArrayEquals( new long[] { 9, 9 }, Intervals.maxAsLongArray( adjustedOverlaps.getB() ) );

		final Pair< Interval, Interval > adjustedOverlapsInFullTile = SubdividedTileOperations.getOverlapsInFullTile( tileBoxPair, adjustedOverlaps );

		Assert.assertArrayEquals( new long[] { 10, 0 }, Intervals.minAsLongArray( adjustedOverlapsInFullTile.getA() ) );
		Assert.assertArrayEquals( new long[] { 17, 3 }, Intervals.maxAsLongArray( adjustedOverlapsInFullTile.getA() ) );

		Assert.assertArrayEquals( new long[] { 12, 16 }, Intervals.minAsLongArray( adjustedOverlapsInFullTile.getB() ) );
		Assert.assertArrayEquals( new long[] { 19, 19 }, Intervals.maxAsLongArray( adjustedOverlapsInFullTile.getB() ) );
	}
}