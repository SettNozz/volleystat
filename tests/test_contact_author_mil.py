import numpy as np

from src.tracking.contact_author_mil import fit_ranker, predict_rankers, ranked_actors


def test_weak_team_loss_can_rank_a_latent_actor_without_person_targets():
    # Only team labels enter fitting. A small synthetic bag has one close
    # candidate on the true side and three noncontact candidates.
    y = np.asarray([0, 1]*10)
    teams = np.tile([0, 0, 1, 1], (len(y), 1))
    x = np.ones((len(y), 4, 2), dtype=np.float32)
    for row, label in enumerate(y):
        x[row, label*2] = [0., 0.]
        x[row, label*2+1] = [.8, .8]
    fitted = fit_ranker(x, teams, y, seed=42, epochs=50)
    probability, shares = predict_rankers([fitted], x, teams)
    assert np.mean((probability >= .5) == y) == 1.
    assert all(shares[i, label*2] > shares[i, label*2+1] for i, label in enumerate(y))
    assert fitted["training_total"] == len(y)
    # Completely missing bags remain neutral rather than inventing an actor.
    empty_probability, empty_shares = predict_rankers([fitted], np.zeros_like(x), np.full_like(teams, -1))
    np.testing.assert_allclose(empty_probability, .5)
    np.testing.assert_allclose(empty_shares, 0)


def test_actor_rank_metadata_preserves_geometry_and_is_not_human_truth():
    actors = [[{"score": .01, "player_id": "P1"}, {"score": .02, "player_id": "P2"}]]
    result = ranked_actors(actors, np.asarray([[.2, .8]]))
    assert result[0][0]["player_id"] == "P2"
    assert result[0][0]["geometric_score"] == .02
    assert actors[0][0]["score"] == .01
