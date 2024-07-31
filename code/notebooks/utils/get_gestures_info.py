
import pandas as pd
def get_gestures_info(turns_info):
   pairs_name = ['pair10', 'pair17', 'pair15', 'pair21', 'pair18', 'pair24', 'pair16', 'pair13', 'pair07', 'pair22', 'pair20',  'pair08', 'pair05', 'pair11', 'pair23', 'pair04', 'pair09', 'pair12', 'pair14' ]
   gestures_info = []
   for pair in pairs_name:
      pair_turns_info = turns_info[pair]
      for turn in pair_turns_info:
         if turn.gesture:
            for gesture in turn.gesture:
               gesture_info = {'pair': pair, 'turn': turn.ID, 'round': turn.round, 'type': gesture.g_type, 'is_gesture': gesture.is_gesture, 'from_ts': gesture.g_from_ts, 'to_ts': gesture.g_to_ts, 'hand': gesture.g_hand, 'referent': gesture.g_referent, 'comment': gesture.g_comment}
               gestures_info.append(gesture_info)
   gestures_info = pd.DataFrame(gestures_info)
   gestures_info['duration'] = gestures_info['to_ts'] - gestures_info['from_ts']
   gestures_info = gestures_info.fillna('')
   gesture_data = gestures_info.groupby('type').count().reset_index().rename(columns={'pair': 'count'})[['type', 'count']]
   gesture_data['normalized_count'] = gesture_data['count'] / gesture_data['count'].sum()
   gesture_data['normalized_count'] = gesture_data['normalized_count'].apply(lambda x: round(x, 2))
   return gestures_info, gesture_data