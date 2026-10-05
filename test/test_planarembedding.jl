using Graphs: edges, ne, nv, src
using ITensorNetworksNext:
    face_coloring, hexagonal_position, leftface, next_edge, planar_embedding, prev_edge
using NamedGraphs: all_edges, named_grid, named_hexagonal_lattice_graph
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
        @test length(emb.positions) == 2 * ne(g)
        for d in all_edges(g)
            @test prev_edge(emb, next_edge(emb, d)) == d
            @test leftface(emb, next_edge(emb, d)) == leftface(emb, d)
        end
        for e in edges(g)
            fs = (leftface(emb, e), leftface(emb, reverse(e)))
            @test count(!isnothing, fs) ≥ 1
        end
        interior = count(
            e -> all(!isnothing, (leftface(emb, e), leftface(emb, reverse(e)))),
            edges(g)
        )
        @test 2 * interior + (ne(g) - interior) == sum(length, emb.faces)

        coloring = face_coloring(emb)
        @test all(
            coloring[f1] != coloring[f2] for f1 in emb.faces, f2 in emb.faces if
                f1 !== f2 && !isdisjoint(src.(f1), src.(f2))
        )
    end
    @test maximum(face_coloring(planar_embedding(named_grid((6, 6)), v -> v))) == 4
    @testset "one outer face required" begin
        g = named_grid((2, 2))
        @test_throws ArgumentError planar_embedding(g, v -> (0.0, 0.0))
    end
end
