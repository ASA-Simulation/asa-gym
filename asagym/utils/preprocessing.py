from typing import List

import asagym.proto.simulator_pb2 as pb


def merge_observations(observations: List[pb.State], cur_sum: pb.Summary) -> pb.Summary:
    new_sum = pb.Summary()
    new_sum.MergeFrom(cur_sum)

    #
    # For now, just replace with the new information
    # TODO: improve this (perhaps state estimation with Kalman Filter)
    #

    for obs in observations:
        if obs.side == pb.BLUE:
            # owner of blue team
            owner_state = obs.owner.player_state
            state = new_sum.own_team.get_or_create(owner_state.id)
            state.MergeFrom(owner_state)
        elif obs.side == pb.RED:
            # owner of red team
            owner_state = obs.owner.player_state
            state = new_sum.ene_team.get_or_create(owner_state.id)
            state.MergeFrom(owner_state)

    # for obs in observations:
    #     # owner
    #     owner_state = obs.owner.player_state
    #     state = new_sum.own_team.get_or_create(owner_state.id)
    #     state.MergeFrom(owner_state)
    #
    #     # wing
    #     wing_state = obs.wing.player_state
    #     state = new_sum.own_team.get_or_create(wing_state.id)
    #     state.MergeFrom(owner_state)
    #
    #     # foes
    #     for foe in obs.foes:
    #         state = new_sum.ene_team.get_or_create(foe.player_state.id)
    #         state.MergeFrom(foe.player_state)

    return new_sum
