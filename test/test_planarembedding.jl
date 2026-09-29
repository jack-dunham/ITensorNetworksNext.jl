using Graphs: edges, ne, nv
using ITensorNetworksNext:
    darts, hexagonal_position, leftface, next_dart, planar_embedding, prev_dart
using NamedGraphs: named_grid, named_hexagonal_lattice_graph
using Test: @test, @test_throws, @testset

@testset "PlanarEmbedding" begin
    @testset "$lattice" for (lattice, g, pos, nfaces, facelength) in (
            ("grid", named_grid((3, 4)), v -> v, 6, 4),
            ("hexagonal", named_hexagonal_lattice_graph(2, 3), hexagonal_position, 6, 6),
        )
        emb = planar_embedding(g, pos)
        @test length(emb.faces) == nfaces
        @test all(f -> length(f) == facelength, emb.faces)
        @test nv(g) - ne(g) + length(emb.faces) == 1
        @test length(darts(emb)) == 2 * ne(g)
        for d in darts(emb)
            @test prev_dart(emb, next_dart(emb, d)) == d
            @test leftface(emb, next_dart(emb, d)) == leftface(emb, d)
        end
        for e in edges(g)
            fs = (leftface(emb, e), leftface(emb, reverse(e)))
            @test count(!iszero, fs) ≥ 1
        end
        interior = count(
            e -> all(!iszero, (leftface(emb, e), leftface(emb, reverse(e)))),
            edges(g)
        )
        @test 2 * interior + (ne(g) - interior) == sum(length, emb.faces)
    end
    @testset "one outer face required" begin
        g = named_grid((2, 2))
        @test_throws ArgumentError planar_embedding(g, v -> (0.0, 0.0))
    end
end
